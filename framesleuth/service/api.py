"""FastAPI surface for Framesleuth local analysis."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import secrets
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import aiohttp
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.background import BackgroundTask

from framesleuth import __version__
from framesleuth.clients.health import get_health_status
from framesleuth.clients.vlm import VLMClient
from framesleuth.config import Settings, get_settings
from framesleuth.errors import (
    FramesleutheException,
    JobCancelledError,
    JobNotFoundError,
    ModelUnavailableError,
    UploadTooLargeError,
    sanitize_message,
)
from framesleuth.jobs.retention import purge_expired_bundles
from framesleuth.jobs.store import JobStore
from framesleuth.logging_config import (
    get_correlation_id,
    get_logger,
    set_correlation_id,
    setup_logging,
)
from framesleuth.orchestrator.graph import AnalysisOrchestrator
from framesleuth.schemas import ContextBundle, JobState
from framesleuth.service import models as m

# Terminal job states — used by the SSE stream to know when to stop.
_TERMINAL_STATES = {JobState.DONE.value, JobState.FAILED.value, JobState.CANCELLED.value}

# A client-supplied X-Request-ID is echoed back and written into every log line for
# that request, so it is only honoured when it is short and unambiguous — otherwise
# a caller could inject newlines/quotes and forge structured log records.
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._\-]{1,64}")

# The only path reachable without the bearer token when API_TOKEN is set.
# Everything else — including /docs, /redoc and /openapi.json — is gated: those
# are outside the /v1 prefix, so the previous `path.startswith("/v1")` test let
# an unauthenticated caller read the full API description.
_PUBLIC_PATHS = frozenset({"/v1/healthz"})


def _error_response(status: int, message: str, code: str, hint: str | None = None) -> JSONResponse:
    """Build the one canonical error envelope, with the request id attached.

    Middleware-generated errors used to emit a flat ``{error, code}`` body while
    route errors emitted FastAPI's ``{"detail": {...}}`` wrapper, so a client had
    to parse two shapes for the same failure class. Everything now goes through
    here (and the exception handlers below), so ``{error, code, hint, request_id}``
    is the single documented shape.
    """
    body: dict[str, Any] = {"error": message, "code": code}
    if hint:
        body["hint"] = hint
    request_id = get_correlation_id()
    if request_id:
        body["request_id"] = request_id
    return JSONResponse(status_code=status, content=body)


logger = get_logger("service.api")

# Content types for stored source recordings, keyed by file suffix.
_VIDEO_MEDIA_TYPES = {
    ".webm": "video/webm",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}


def _find_source_video(bundle_dir: Path) -> Path | None:
    """Return the stored ``source.*`` recording in ``bundle_dir``, if present."""
    for source in sorted(bundle_dir.glob("source.*")):
        if source.suffix.lower() in _VIDEO_MEDIA_TYPES:
            return source
    return None


def _safe_json(raw: str | None) -> Any:
    """Parse a JSON form field, tolerating absent or malformed input."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _require_json(raw: str | None, field: str) -> Any:
    """Parse a JSON form field, raising 400 when it is present but malformed.

    Silently dropping unparseable input turned a caller's bug (a mis-encoded
    sidecar payload) into a mysteriously empty bundle. Absent/blank is fine
    (the field is optional); present-but-invalid is a client error worth naming.
    """
    if raw is None or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": f"{field} is not valid JSON", "code": f"bad_{field}"},
        ) from exc


def _require_video_magic(head: bytes, suffix: str) -> None:
    """Reject a payload whose first bytes are not a known video container.

    The stored suffix comes from the *client-supplied* filename, so extension
    alone proves nothing: ``evil.mp4`` containing a zip was accepted, written to
    disk in full, hashed and given a job row, and only rejected later inside the
    pipeline. Sniffing the first chunk moves that rejection to ingest, before any
    of the expensive work.

    Deliberately permissive — this is a sanity gate, not a validator; PyAV
    remains the authority on whether the stream actually decodes.
    """
    if len(head) < 12:
        raise HTTPException(
            status_code=422,
            detail={"error": "Upload is too small to be a video", "code": "unsupported_media"},
        )
    ok = (
        head[4:8] == b"ftyp"  # ISO-BMFF: mp4 / mov / m4v
        or head[:4] == b"\x1a\x45\xdf\xa3"  # EBML: webm / mkv
        or (head[:4] == b"RIFF" and head[8:12] == b"AVI ")  # avi
        or head[:4] == b"\x00\x00\x01\xba"  # MPEG program stream
        or head[:3] == b"FLV"
    )
    if not ok:
        raise HTTPException(
            status_code=415,
            detail={
                "error": "Upload does not look like a video container",
                "code": "unsupported_media",
                "hint": f"The bytes do not match a {suffix} container. "
                "Re-export as H.264 MP4 or VP9 WebM.",
            },
        )


def _resolve_workspace_root(repo_root: str | None, settings: Settings) -> Path | None:
    """Validate and confine the optional code-grounding root.

    Grounding reads every source file under this path and embeds matching lines
    in the returned bundle, so an unconstrained value turns the API into a file
    reader for the whole host. ``GROUNDING_ROOTS`` is the allowlist: while it is
    empty (the default) the HTTP surface refuses ``repo_root`` outright, because
    a browser origin is allowed to call this API and must not be able to point
    the scan wherever it likes. The MCP path is unaffected — there the caller is
    the local editor, already trusted with the workspace.
    """
    if not repo_root or not repo_root.strip():
        return None
    allowed = settings.grounding_roots_list
    if not allowed:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Code grounding over HTTP is not enabled",
                "code": "grounding_disabled",
                "hint": "Set GROUNDING_ROOTS to the directories this agent may scan.",
            },
        )
    candidate = Path(repo_root).expanduser()
    if not candidate.is_dir():
        raise HTTPException(
            status_code=400,
            detail={"error": "repo_root is not a directory", "code": "bad_repo_root"},
        )
    resolved = candidate.resolve()
    for root in allowed:
        if resolved == root or root in resolved.parents:
            return resolved
    raise HTTPException(
        status_code=403,
        detail={
            "error": "repo_root is outside the allowed grounding roots",
            "code": "grounding_forbidden",
        },
    )


def _read_metrics(bundle_path: str | None) -> dict[str, Any] | None:
    """Return the per-stage ``metrics.json`` next to a bundle, if present.

    Surfaces stage timings and the degraded-stage list to a polling client so the
    pipeline's progress is observable beyond the coarse ``progress_pct``.
    """
    if not bundle_path:
        return None
    metrics_path = Path(bundle_path).parent / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        loaded: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
        return loaded
    except (OSError, json.JSONDecodeError):
        return None


class AppState:
    """State container for app dependencies."""

    def __init__(self, settings: Settings) -> None:
        """Build the store, VLM client, and orchestrator from ``settings``."""
        self.settings = settings
        self.store = JobStore(settings.DATABASE_PATH)
        self.vlm_client = VLMClient.from_settings(settings)
        self.orchestrator = AnalysisOrchestrator(settings, self.store, self.vlm_client)
        # Bounds how many analyses run the heavy pipeline at once. Excess jobs sit
        # in QUEUED state (the analyze response already returned their id), giving
        # real backpressure instead of fanning out unbounded VLM/ffmpeg work.
        self._job_semaphore = asyncio.Semaphore(max(1, settings.MAX_CONCURRENT_JOBS))
        # Every in-flight analysis, so shutdown can drain them. With Starlette's
        # BackgroundTasks the pipeline ran *inside the HTTP request task*, which
        # meant a 30-minute job either blocked shutdown or was killed mid-write,
        # leaving a half-written bundle directory and a job row stuck non-terminal
        # until the next boot's orphan sweep.
        self._jobs: set[asyncio.Task[None]] = set()
        self._accepting = True

    def reopen(self) -> None:
        """Accept work again (called on startup, after a previous drain)."""
        self._accepting = True

    def submit(self, coro: Any) -> bool:
        """Schedule an analysis on the app's own task set.

        Returns ``False`` once shutdown has begun, so a request that races the
        drain is rejected with 503 instead of being silently dropped.
        """
        if not self._accepting:
            coro.close()  # never leave an un-awaited coroutine behind
            return False
        task = asyncio.create_task(coro)
        self._jobs.add(task)
        task.add_done_callback(self._jobs.discard)
        return True

    async def drain(self, timeout_s: float) -> None:
        """Stop accepting work and let in-flight analyses finish, then give up.

        Anything still running past the budget is cancelled and recorded as a
        terminal failure, so a restart never leaves a job that a poller would
        wait on forever.
        """
        self._accepting = False
        if not self._jobs:
            return
        pending = set(self._jobs)
        logger.info("Draining %d in-flight job(s), up to %.0fs", len(pending), timeout_s)
        done, still_running = await asyncio.wait(pending, timeout=max(0.0, timeout_s))
        for task in still_running:
            task.cancel()
        if still_running:
            await asyncio.gather(*still_running, return_exceptions=True)
            logger.warning("Cancelled %d job(s) that outlived the drain window", len(still_running))
        logger.info("Drain complete (%d finished cleanly)", len(done))

    async def run_job(
        self,
        job_id: str,
        temp_path: Path,
        source_video: str,
        *,
        sidecars: Any,
        capture_options: str | None,
        intent: str | None,
        skill: str | None,
        system_prompt: str | None,
        action: str | None,
        action_prompt: str | None,
        workspace_root: Path | None = None,
    ) -> None:
        """Run one analysis to completion in the background, bounded by the semaphore.

        The orchestrator marks the job FAILED on unexpected errors; this wrapper
        additionally records typed ``FramesleutheException``s (e.g. unsupported
        media) so a poller sees a structured ``error`` rather than a stuck job,
        and always removes the temp upload.
        """
        async with self._job_semaphore:
            try:
                await self._run_with_retries(
                    job_id,
                    temp_path,
                    source_video,
                    sidecars=sidecars,
                    # The orchestrator's parameter is `user_intent`; the HTTP
                    # form field is `intent`. Keep the rename here, at the seam.
                    user_intent=intent,
                    skill=skill,
                    system_prompt=system_prompt,
                    action=action,
                    action_prompt=action_prompt,
                    workspace_root=workspace_root,
                )
                if capture_options:
                    persisted = await self.store.get_job(job_id)
                    if persisted and persisted.bundle_path:
                        target = Path(persisted.bundle_path).parent / "capture_options.json"
                        await asyncio.to_thread(target.write_text, capture_options, "utf-8")
            except JobCancelledError:
                # The orchestrator already recorded the terminal CANCELLED state.
                logger.info("Background analysis cancelled for job %s", job_id)
            except asyncio.CancelledError:
                # Shutdown drain gave up on us. Record a terminal state now — an
                # abandoned non-terminal row would otherwise strand every poller
                # (and the content-hash idempotency) until the next boot.
                await self.store.finalize(
                    job_id,
                    state=JobState.FAILED,
                    error_json={
                        "error": "Server shut down while the analysis was running",
                        "code": "interrupted",
                    },
                )
                raise
            except FramesleutheException as exc:
                await self.store.finalize(job_id, state=JobState.FAILED, error_json=exc.to_dict())
            except Exception as exc:
                logger.exception("Background analysis failed for job %s", job_id)
                # The orchestrator normally records FAILED itself, but relying on
                # that leaves a job stranded non-terminal forever if it dies
                # before it gets there. `finalize` is guarded, so this is a no-op
                # when a terminal state was already written.
                await self.store.finalize(
                    job_id,
                    state=JobState.FAILED,
                    error_json={
                        "error": sanitize_message(str(exc)) or "Analysis failed",
                        "code": "internal_error",
                    },
                )
            finally:
                temp_path.unlink(missing_ok=True)
                await self._notify_webhook(job_id)

    async def _run_with_retries(
        self,
        job_id: str,
        temp_path: Path,
        source_video: str,
        **kwargs: Any,
    ) -> None:
        """Run the pipeline, retrying only failures that a retry can actually fix.

        A local model server that is still loading a model refuses connections or
        returns 503 for the first few seconds — previously that cost the user the
        whole upload, which they then had to re-send. Transient failures
        (model unavailable, watchdog timeout) get a bounded number of retries with
        exponential backoff; deterministic ones (unsupported media, duration
        exceeded) are terminal on the first attempt because a retry would fail
        identically.
        """
        attempts = max(1, self.settings.JOB_MAX_ATTEMPTS)
        for attempt in range(1, attempts + 1):
            try:
                # JOB_TIMEOUT_S is a hard watchdog: a wedged model server must
                # not hold a pipeline slot (and the caller's poll loop) forever.
                async with asyncio.timeout(self.settings.JOB_TIMEOUT_S):
                    await self.orchestrator.run(job_id, temp_path, source_video, **kwargs)
                return
            except TimeoutError as exc:
                last: Exception = exc
                logger.error(
                    "Job %s exceeded JOB_TIMEOUT_S (attempt %d/%d)", job_id, attempt, attempts
                )
                error = {
                    "error": f"Analysis exceeded {self.settings.JOB_TIMEOUT_S}s",
                    "code": "job_timeout",
                }
            except ModelUnavailableError as exc:
                last = exc
                logger.warning(
                    "Job %s: model unavailable (attempt %d/%d)", job_id, attempt, attempts
                )
                error = exc.to_dict()

            if attempt >= attempts or not await self._can_retry(job_id, temp_path):
                await self.store.finalize(job_id, state=JobState.FAILED, error_json=error)
                logger.error("Job %s failed permanently: %s", job_id, last)
                return
            # Reset to QUEUED so a poller sees "waiting to run again", not a
            # phantom mid-pipeline stage from the attempt that just died.
            await self.store.update_job(job_id, state=JobState.QUEUED, progress_pct=0)
            await asyncio.sleep(min(30.0, 2.0**attempt))

    async def _can_retry(self, job_id: str, temp_path: Path) -> bool:
        """Whether another attempt is worth making (upload intact, not cancelled)."""
        if not temp_path.exists():
            return False
        return not await self.store.is_cancel_requested(job_id)

    async def _notify_webhook(self, job_id: str) -> None:
        """POST a compact completion payload to ``WEBHOOK_URL`` (best-effort).

        Fires once a job reaches a terminal state so an external system can react
        without polling. Never raises — a webhook failure must not affect the job.
        """
        url = self.settings.WEBHOOK_URL.strip()
        if not url:
            return
        job = await self.store.get_job(job_id)
        if job is None or job.state.value not in _TERMINAL_STATES:
            return
        payload: dict[str, Any] = {
            "id": job.id,
            "state": job.state.value,
            "source_video": job.source_video,
            "error": job.error_json,
        }
        if job.bundle_path and Path(job.bundle_path).exists():
            try:
                bundle = json.loads(Path(job.bundle_path).read_text(encoding="utf-8"))
                payload["title"] = bundle.get("title")
                payload["action"] = bundle.get("action")
            except (OSError, json.JSONDecodeError):
                pass
        try:
            timeout = aiohttp.ClientTimeout(total=self.settings.WEBHOOK_TIMEOUT_S)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                await session.post(url, json=payload)
            logger.info("Webhook delivered for job %s (%s)", job_id, job.state.value)
        except Exception as exc:  # webhook is best-effort; never affect the job
            logger.warning("Webhook POST failed for job %s: %s", job_id, exc)


def create_app(settings: Settings | None = None) -> FastAPI:  # noqa: C901
    """Create configured FastAPI application.

    Complexity is inherent to an app factory that registers every route as a
    nested closure over shared state; the individual handlers stay simple.
    """
    settings = settings or get_settings()
    # Install the structured (JSON) log handler before anything else logs.
    # Without this, LOG_LEVEL/LOG_JSON were inert: the `framesleuth` logger had
    # no handler of its own, so every module's carefully-structured log line fell
    # through to the root logger's default plain formatter (or was dropped).
    setup_logging(level=settings.LOG_LEVEL, json_format=settings.LOG_JSON)
    state = AppState(settings)

    async def _retention_loop() -> None:
        """Re-run the retention sweep periodically so a long-lived server purges too."""
        interval = max(60.0, settings.RETENTION_SWEEP_INTERVAL_S)
        while True:
            await asyncio.sleep(interval)
            try:
                await purge_expired_bundles(
                    state.store, settings.BUNDLE_DIR, settings.BUNDLE_TTL_DAYS
                )
            except Exception:  # the sweep must never kill the server
                logger.exception("Retention sweep failed")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        settings.validate_paths()
        # Re-arm the submission gate: `drain()` closes it on shutdown, and the
        # AppState outlives a single lifespan (it is built once in create_app),
        # so an app that is started again — the normal case under a test client,
        # and possible under a reloading server — would otherwise refuse every
        # job forever with 503.
        state.reopen()
        await state.store.initialize()
        # Crash/restart recovery: any job still non-terminal in the DB belonged
        # to a previous process and can never finish — fail it now so queue
        # depth is honest and idempotency never returns a zombie.
        orphaned = await state.store.fail_orphaned_jobs()
        if orphaned:
            logger.warning("Failed %d orphaned job(s) from a previous run", orphaned)
        # Uploads whose background task never ran (crash between accept and run)
        # would otherwise accumulate forever next to the bundles.
        for stale_upload in settings.BUNDLE_DIR.glob("upload-*"):
            stale_upload.unlink(missing_ok=True)
        # Retention sweep: drop bundles/jobs past the TTL so disk stays bounded —
        # once now, then periodically while the server runs.
        await purge_expired_bundles(state.store, settings.BUNDLE_DIR, settings.BUNDLE_TTL_DAYS)
        sweeper = asyncio.create_task(_retention_loop())
        try:
            yield
        finally:
            sweeper.cancel()
            with suppress(asyncio.CancelledError):
                await sweeper
            # Let in-flight analyses finish (bounded), then cancel the rest and
            # record a terminal state for them, so a restart never strands a job
            # that pollers are waiting on.
            await state.drain(settings.SHUTDOWN_DRAIN_S)
            # Release the pooled VLM HTTP session on shutdown.
            await state.vlm_client.aclose()

    app = FastAPI(
        title="Framesleuth API",
        version=__version__,
        summary="Local video → structured Context Bundle, for coding agents.",
        lifespan=lifespan,
        # Every route declares response models, so this schema is a complete
        # machine description of the API — `scripts/export_openapi.py` writes it
        # out and the website generates its typed client from it.
        openapi_tags=[
            {"name": "meta", "description": "Health, version, and capability discovery."},
            {"name": "catalog", "description": "Selectable summary skills and action modes."},
            {"name": "jobs", "description": "Submit an analysis and follow its lifecycle."},
            {"name": "reports", "description": "Read a finished Context Bundle and its artifacts."},
            {"name": "render", "description": "Turn an HTML animation into a video clip."},
        ],
    )
    app.state.app_state = state

    # Optional bearer-token auth. CORS only controls what a browser may *read*;
    # it does not stop a DNS-rebinding page or another local process from
    # *sending* requests to this loopback API. Registered before the CORS
    # middleware so CORS wraps it and preflights (OPTIONS) never need the token.
    if settings.API_TOKEN:
        expected_auth = f"Bearer {settings.API_TOKEN}".encode()

        @app.middleware("http")
        async def require_api_token(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            if request.url.path not in _PUBLIC_PATHS:
                # Starlette decodes header bytes as latin-1, so a header carrying
                # any byte > 0x7F yields a non-ASCII str — and `compare_digest` on
                # str raises TypeError for those, which turned a bad credential
                # into an unhandled 500. Compare *bytes* so every input is a clean
                # 401, and the comparison stays constant-time.
                provided = request.headers.get("authorization", "").encode("latin-1", "replace")
                if not secrets.compare_digest(provided, expected_auth):
                    return _error_response(401, "Missing or invalid API token", "unauthorized")
            return await call_next(request)

    # Reject clearly-oversized request bodies from the declared Content-Length,
    # before any multipart parsing or disk spooling happens. The streaming read
    # in /v1/analyze remains the authoritative cap (Content-Length can be absent
    # or wrong); this just refuses the obvious cases as cheaply as possible.
    max_body_bytes = (settings.MAX_UPLOAD_MB + 8) * 1024 * 1024  # slack for multipart framing

    @app.middleware("http")
    async def reject_oversized_bodies(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > max_body_bytes:
            return _error_response(
                413,
                f"Request body exceeds {settings.MAX_UPLOAD_MB} MB",
                "upload_too_large",
                f"Record a shorter video (limit: {settings.MAX_UPLOAD_MB} MB).",
            )
        return await call_next(request)

    # Correlate every request/log line. Registered after the body guard so it runs
    # *outside* it: even a rejected oversized upload is traceable. The id is echoed
    # on the response so a client (or the website) can quote it in a bug report and
    # find the exact server-side log lines for that call.
    @app.middleware("http")
    async def correlate_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Honour a caller-supplied id, but never trust it verbatim in a log line:
        # cap the length and keep it to safe characters so it cannot forge fields.
        incoming = request.headers.get("x-request-id", "")[:64]
        request_id = incoming if _SAFE_REQUEST_ID.fullmatch(incoming) else None
        request_id = set_correlation_id(request_id)
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    # One error envelope for the whole API. Previously a route error came back as
    # FastAPI's `{"detail": {...}}` while a middleware error was a flat
    # `{"error": ...}`, and an unhandled exception was `{"detail": "Internal
    # Server Error"}` with no code at all — three shapes a client had to parse for
    # the same failure class.
    @app.exception_handler(FramesleutheException)
    async def _handle_domain_error(_r: Request, exc: FramesleutheException) -> JSONResponse:
        return _error_response(exc.status_code, exc.message, exc.code.value, exc.hint)

    @app.exception_handler(HTTPException)
    async def _handle_http_error(_r: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict):
            return _error_response(
                exc.status_code,
                str(detail.get("error", "Request failed")),
                str(detail.get("code", "http_error")),
                detail.get("hint"),
            )
        return _error_response(exc.status_code, str(detail), f"http_{exc.status_code}")

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(_r: Request, exc: RequestValidationError) -> JSONResponse:
        # Field errors are useful to a developer but must not echo submitted
        # values back (a form field can carry user content), so only the location
        # and the rule that failed are surfaced.
        fields = [
            {"field": ".".join(str(p) for p in err.get("loc", ())), "reason": err.get("msg", "")}
            for err in exc.errors()[:10]
        ]
        response = _error_response(422, "Request validation failed", "validation_error")
        body = json.loads(bytes(response.body))
        body["fields"] = fields
        return JSONResponse(status_code=422, content=body)

    @app.exception_handler(Exception)
    async def _handle_unexpected(_r: Request, exc: Exception) -> JSONResponse:
        # The full traceback goes to the log; the response carries only the
        # request id, so the local filesystem layout never leaks to a browser
        # origin (same rationale as `sanitize_message`).
        logger.exception("Unhandled error: %s", type(exc).__name__)
        return _error_response(500, "Internal server error", "internal_error")

    # Allow the configured capture-extension origins *and* whatever
    # `web_origins_list` resolves to: loopback dev servers (unless
    # ALLOW_LOCAL_DEV_ORIGINS is off) plus anything the operator opted into via
    # WEB_ORIGINS. The hosted site is NOT trusted by default — this is the
    # open-source, runs-on-your-machine half of the product, and shipping a
    # remote origin in the default allowlist contradicted that.
    # Origins are an exact allowlist — never a pattern: every Chrome
    # extension ID is 32 chars of [a-p], so a `chrome-extension://[a-p]{32}`
    # regex would open the API to *every* extension the user has installed.
    #
    # ``allow_private_network=True`` answers Chrome's Private Network Access
    # preflight: a page on a *public* HTTPS origin (e.g. https://framesleuth.com)
    # calling this loopback API gets ``Access-Control-Request-Private-Network: true``
    # honored with the matching allow header, so the request isn't blocked.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[*settings.chrome_extension_origins, *settings.web_origins_list],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        # Without this the browser hides `X-Request-ID` from the page, so the
        # website could never show the id a user needs to quote in a bug report.
        expose_headers=["X-Request-ID"],
        allow_private_network=True,
    )

    @app.get("/v1/healthz", tags=["meta"], response_model=m.HealthResponse)
    async def healthz() -> dict[str, object]:
        """Report dependency readiness and the live queue depth.

        Public even when ``API_TOKEN`` is set, so a client can tell "backend is
        down" from "backend needs a token" before authenticating.
        """
        health = await get_health_status(
            vlm_url=settings.VLM_URL,
            coder_url=settings.CODER_URL,
            bundle_dir=str(settings.BUNDLE_DIR),
            queue_depth=await state.store.count_active(),
            # A reachable server with the model unpulled still answers 200. The
            # README treats `vlm: ready` as "recordings analyze with a real
            # classification", so the model itself has to be confirmed.
            vlm_model=settings.VLM_MODEL,
        )
        payload = health.model_dump()
        # Surface optional HTML→video readiness so it's diagnosable in the running
        # process (cheap, no browser launch).
        from framesleuth.pipeline.html_render import render_availability

        render = dict(render_availability())
        # ALWAYS strip the verbose diagnostics. `python`/`hint` embed the server's
        # absolute interpreter path — the deployment-layout leak `sanitize_message`
        # exists to prevent for job errors — and this endpoint is in _PUBLIC_PATHS,
        # so it is reachable without a token no matter how the agent is configured.
        # Gating them on `API_TOKEN` was exactly backwards: it exposed the path
        # precisely when the operator had asked for authentication.
        # The full diagnostics live on the token-gated /v1/version instead.
        render.pop("python", None)
        render.pop("hint", None)
        payload["render"] = render
        return payload

    @app.get("/v1/version", tags=["meta"], response_model=m.VersionResponse)
    async def version() -> m.VersionResponse:
        """Report the build and which optional capabilities this install has.

        Lets a client feature-detect once instead of probing endpoints and
        guessing whether a 404/503 means "unsupported" or "broken".
        """
        from framesleuth.pipeline.html_render import render_availability

        availability = render_availability()
        return m.VersionResponse(
            version=__version__,
            schema_version=ContextBundle.model_fields["schema_version"].default,
            # The verbose render diagnostics (interpreter path, the actionable
            # "what is missing" hint) belong here rather than on /v1/healthz,
            # because this endpoint IS covered by the token gate.
            render=m.RenderAvailability(**availability),
            capabilities={
                "render_html": bool(availability.get("ready")),
                "ocr_backstop": settings.OCR_BACKSTOP,
                "auth_required": bool(settings.API_TOKEN),
                "webhook": bool(settings.WEBHOOK_URL.strip()),
                "redact_pii": settings.REDACT_PII,
                "grounding_over_http": bool(settings.grounding_roots_list),
            },
        )

    @app.get("/v1/skills", tags=["catalog"], response_model=m.SkillsResponse)
    async def skills() -> dict[str, object]:
        """List built-in summary skills the caller can pass to ``/v1/analyze``."""
        from framesleuth.skills import DEFAULT_SKILL, list_skills

        return {"default": DEFAULT_SKILL, "skills": list_skills()}

    @app.get("/v1/actions", tags=["catalog"], response_model=m.ActionsResponse)
    async def actions() -> dict[str, object]:
        """List built-in action modes the caller can pass to ``/v1/analyze``."""
        from framesleuth.actions import DEFAULT_ACTION, list_actions

        return {"default": DEFAULT_ACTION, "auto": True, "actions": list_actions()}

    @app.post(
        "/v1/analyze",
        status_code=202,
        tags=["jobs"],
        response_model=m.AnalyzeAccepted,
        responses={400: m.ERROR_422, 413: m.ERROR_413, 415: m.ERROR_422, 429: m.ERROR_409},
    )
    async def analyze(
        video: UploadFile = File(..., description="The recording (mp4/webm/mkv/mov/avi)"),
        sidecars: str | None = Form(default=None),
        capture_options: str | None = Form(default=None),
        intent: str | None = Form(default=None),
        skill: str | None = Form(default=None),
        system_prompt: str | None = Form(default=None),
        action: str | None = Form(default=None),
        action_prompt: str | None = Form(default=None),
        repo_root: str | None = Form(default=None),
    ) -> m.AnalyzeAccepted:
        """Accept a recording and queue it for analysis.

        Returns ``202`` with the new ``job_id`` immediately; the analysis runs in
        the background (bounded by ``MAX_CONCURRENT_JOBS``). Poll ``/v1/jobs/{id}``
        for progress and ``/v1/report/{id}`` once the state is ``done``.
        """
        # Everything that can be rejected *without* touching disk happens first,
        # so a bad request never leaves a half-gigabyte temp file behind. The
        # `repo_root` check in particular used to run after the streaming write
        # and raise past its cleanup, making "post a video with a bad repo_root"
        # a one-line disk-fill.
        parsed_sidecars = _require_json(sidecars, "sidecars")
        filename = video.filename or "upload.mp4"
        suffix = Path(filename).suffix.lower()
        if suffix not in _VIDEO_MEDIA_TYPES:
            raise HTTPException(
                status_code=415,
                detail={
                    "error": f"Unsupported file type {suffix or '(none)'}",
                    "code": "unsupported_media",
                    "hint": f"Supported: {', '.join(sorted(_VIDEO_MEDIA_TYPES))}",
                },
            )
        workspace_root = _resolve_workspace_root(repo_root, settings)

        # Backpressure: MAX_CONCURRENT_JOBS bounds *execution*, not the queue, so
        # without this a burst of uploads accumulates rows and 512 MB temp files
        # faster than the pipeline drains them.
        if await state.store.count_active() >= settings.MAX_QUEUED_JOBS:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Too many jobs are already queued",
                    "code": "queue_full",
                    "hint": "Wait for an in-flight analysis to finish, then retry.",
                },
            )

        temp_path = settings.BUNDLE_DIR / f"upload-{uuid.uuid4()}{suffix}"
        hasher = hashlib.sha256()
        received = 0
        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        try:
            # Stream the upload to disk in chunks — never buffer the whole video
            # in RAM — and enforce the size limit *while* reading, so an
            # oversized body is rejected as soon as it crosses the cap.
            with temp_path.open("wb") as sink:
                first = True
                while chunk := await video.read(1024 * 1024):
                    if first:
                        # Sniff the container magic on the first chunk. The
                        # extension is client-supplied, so without this the API
                        # happily wrote, hashed and queued 512 MB of arbitrary
                        # bytes and only rejected them minutes later, inside the
                        # pipeline, long after the 202.
                        _require_video_magic(chunk, suffix)
                        first = False
                    received += len(chunk)
                    if received > max_bytes:
                        raise UploadTooLargeError(received / (1024 * 1024), settings.MAX_UPLOAD_MB)
                    hasher.update(chunk)
                    await asyncio.to_thread(sink.write, chunk)

            # Idempotency: identical bytes *and* identical analysis options map to
            # the same key. The sidecars are hashed by CONTENT, not by presence —
            # `bool(sidecars)` collapsed every distinct console/network/click
            # payload to True, so a second caller uploading the same recording
            # with different sidecars silently received the first caller's
            # analysis, built from someone else's data.
            hasher.update(
                json.dumps(
                    {
                        "intent": intent,
                        "skill": skill,
                        "system_prompt": system_prompt,
                        "action": action,
                        "action_prompt": action_prompt,
                        "repo_root": str(workspace_root) if workspace_root else None,
                        "sidecars": parsed_sidecars,
                        "capture_options": capture_options,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )

            job_id = str(uuid.uuid4())
            job, created = await state.store.claim_job(job_id, hasher.hexdigest(), filename)
            if not created:
                temp_path.unlink(missing_ok=True)
                return m.AnalyzeAccepted(job_id=job.id, status=job.state, idempotent=True)

            # Hand off to the app's own task set (not Starlette BackgroundTasks):
            # the pipeline can run for 30 minutes and must be drainable on
            # shutdown rather than pinned to this HTTP request's task.
            accepted = state.submit(
                state.run_job(
                    job_id,
                    temp_path,
                    filename,
                    sidecars=parsed_sidecars,
                    capture_options=capture_options,
                    intent=intent,
                    skill=skill,
                    system_prompt=system_prompt,
                    action=action,
                    action_prompt=action_prompt,
                    workspace_root=workspace_root,
                )
            )
            if not accepted:
                await state.store.finalize(
                    job_id,
                    state=JobState.FAILED,
                    error_json={"error": "Server is shutting down", "code": "shutting_down"},
                )
                raise HTTPException(
                    status_code=503,
                    detail={"error": "Server is shutting down", "code": "shutting_down"},
                )
            return m.AnalyzeAccepted(job_id=job_id, status=JobState.QUEUED, idempotent=False)
        except BaseException:
            # One cleanup path covering every failure after the file was opened.
            temp_path.unlink(missing_ok=True)
            raise

    @app.get("/v1/jobs", tags=["jobs"], response_model=m.JobListResponse)
    async def list_jobs(
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        state_filter: list[JobState]
        | None = Query(
            default=None,
            alias="state",
            description="Repeatable. Only return jobs in these states.",
        ),
    ) -> m.JobListResponse:
        """List jobs, newest first, with pagination and state filtering.

        Lets a client browse analysis history — and ask for just the failed or
        just the running ones — instead of paging everything and filtering
        client-side.
        """
        jobs, total = await state.store.list_jobs_page(
            limit=limit, offset=offset, states=state_filter
        )
        return m.JobListResponse(
            total=total,
            limit=limit,
            offset=offset,
            jobs=[
                m.JobSummary(
                    id=job.id,
                    state=job.state,
                    progress_pct=job.progress_pct,
                    source_video=job.source_video,
                    created_at=job.created_at,
                    updated_at=job.updated_at,
                )
                for job in jobs
            ],
        )

    @app.get(
        "/v1/jobs/{job_id}", tags=["jobs"], response_model=m.JobDetail, responses={404: m.ERROR_404}
    )
    async def get_job(job_id: str) -> m.JobDetail:
        """Return one job's lifecycle state, progress, and per-stage metrics."""
        job = await state.store.get_job(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return m.JobDetail(
            id=job.id,
            state=job.state,
            progress_pct=job.progress_pct,
            source_video=job.source_video,
            # Relative to the bundle root — the absolute filesystem path stays
            # server-side (responses are readable by allowlisted web origins).
            bundle_path=f"{job.id}/bundle.json" if job.bundle_path else None,
            error=job.error_json,
            metrics=await asyncio.to_thread(_read_metrics, job.bundle_path),
            cancel_requested=job.cancel_requested,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )

    @app.delete(
        "/v1/jobs/{job_id}",
        tags=["jobs"],
        response_model=m.JobCancelResponse,
        responses={404: m.ERROR_404, 409: m.ERROR_409},
    )
    async def cancel_job(job_id: str) -> m.JobCancelResponse:
        """Request cooperative cancellation of a running/queued job.

        The pipeline checks the cancel flag at each stage boundary and transitions
        to ``cancelled``. Already-terminal jobs are returned unchanged (``409``).
        """
        job = await state.store.get_job(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        if job.state.value in _TERMINAL_STATES:
            raise HTTPException(
                status_code=409,
                detail={"error": f"Job already {job.state.value}", "code": "not_cancellable"},
            )
        await state.store.request_cancel(job_id)
        return m.JobCancelResponse(id=job_id, state=job.state, cancel_requested=True)

    @app.get("/v1/jobs/{job_id}/events", tags=["jobs"], responses={404: m.ERROR_404})
    async def job_events(job_id: str, request: Request) -> StreamingResponse:
        """Stream job progress as Server-Sent Events until a terminal state.

        Each event is a JSON snapshot (`state`, `progress_pct`); the stream closes
        once the job is `done`/`failed`/`cancelled`. Lets a client follow progress
        without polling.
        """
        job = await state.store.get_job(job_id)
        if job is None:
            raise JobNotFoundError(job_id)

        async def _stream() -> AsyncIterator[bytes]:
            # Bounded so a stuck job can never hold the connection forever
            # (JOB_TIMEOUT_S at the poll interval), with a terminal early-exit.
            poll_s = 0.5
            heartbeat_every = int(15 / poll_s)
            last_payload: str | None = None
            for tick in range(int(settings.JOB_TIMEOUT_S / poll_s)):
                # Without this the generator kept querying SQLite twice a second
                # for up to JOB_TIMEOUT_S after the client had already gone away.
                if await request.is_disconnected():
                    return
                current = await state.store.get_job(job_id)
                if current is None:
                    # The row vanished mid-stream (e.g. retention). Tell the
                    # client explicitly instead of a silent close it can't read.
                    yield b'event: gone\ndata: {"code": "job_deleted"}\n\n'
                    return
                payload = json.dumps(
                    {
                        "id": current.id,
                        "state": current.state.value,
                        "progress_pct": current.progress_pct,
                    }
                )
                if payload != last_payload:
                    last_payload = payload
                    yield f"id: {tick}\ndata: {payload}\n\n".encode()
                elif tick % heartbeat_every == 0:
                    # A comment frame keeps idle proxies and browsers from
                    # dropping a connection during a long, quiet VLM stage.
                    yield b": keepalive\n\n"
                if current.state.value in _TERMINAL_STATES:
                    return
                await asyncio.sleep(poll_s)
            # Budget exhausted without reaching a terminal state: emit an explicit
            # timeout frame so the client can distinguish this from completion.
            yield b'event: timeout\ndata: {"code": "stream_timeout"}\n\n'

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                # Stops nginx and friends from buffering the stream into silence.
                "X-Accel-Buffering": "no",
            },
        )

    async def _load_bundle(job_id: str) -> dict[str, Any]:
        """Read a finished bundle, mapping every not-ready case to a typed error."""
        job = await state.store.get_job(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        if not job.bundle_path:
            raise HTTPException(
                status_code=409,
                detail={"error": "Report not ready", "code": "not_ready"},
            )
        bundle_path = Path(job.bundle_path)
        if not bundle_path.exists():
            raise HTTPException(
                status_code=404,
                detail={"error": "Report missing", "code": "missing_bundle"},
            )
        # Bundles routinely run to hundreds of KB; parsing one on the event loop
        # stalls every other request (health, polling, SSE) for its duration.
        raw = await asyncio.to_thread(bundle_path.read_text, "utf-8")
        parsed: dict[str, Any] = json.loads(raw)
        return parsed

    @app.get(
        "/v1/report/{job_id}",
        tags=["reports"],
        # ContextBundle is declared for the *schema* (so codegen produces a real
        # typed client) but NOT as `response_model`, which would re-validate on
        # the way out: a bundle written by an older schema version would then
        # 500 instead of being readable. The file on disk is the source of truth.
        response_model=None,
        responses={
            200: {"model": ContextBundle, "description": "The Context Bundle"},
            404: m.ERROR_404,
            409: m.ERROR_409,
        },
    )
    async def get_report(job_id: str) -> Any:
        """Return the full Context Bundle for a finished job."""
        return await _load_bundle(job_id)

    @app.get(
        "/v1/report/{job_id}/render",
        tags=["reports"],
        response_model=m.RenderedArtifact,
        responses={404: m.ERROR_404, 409: m.ERROR_409, 422: m.ERROR_422},
    )
    async def render_report(
        job_id: str,
        format: str = Query(default="markdown", description="markdown | issue | test-plan"),
        download: bool = Query(default=False, description="Return raw text as a file download"),
    ) -> Any:
        """Project a finished bundle into a shareable artifact.

        Reuses exactly the renderers the MCP ``render`` tool calls, so the two
        surfaces cannot drift — previously only MCP clients could get markdown /
        issue / test-plan output and a web client had to re-implement it.
        """
        from framesleuth.render import RENDER_FORMATS
        from framesleuth.render import render as render_bundle

        key = format.strip().lower()
        if key not in RENDER_FORMATS:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": f"Unknown render format {format!r}",
                    "code": "bad_format",
                    "hint": f"Expected one of: {', '.join(RENDER_FORMATS)}",
                },
            )
        text = render_bundle(await _load_bundle(job_id), key)
        if download:
            suffix = "md" if key != "issue" else "issue.md"
            return PlainTextResponse(
                text,
                media_type="text/markdown; charset=utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="framesleuth-{job_id}.{suffix}"'
                },
            )
        return m.RenderedArtifact(id=job_id, format=key, text=text)  # type: ignore[arg-type]

    @app.get(
        "/v1/report/{job_id}/prompt",
        tags=["reports"],
        response_model=m.FixPromptResponse,
        responses={404: m.ERROR_404, 409: m.ERROR_409},
    )
    async def report_prompt(job_id: str) -> m.FixPromptResponse:
        """Return the grounded, evidence-only action prompt for a report.

        The same text the MCP ``fix-prompt`` resource serves — so a web client can
        hand it to whichever coding agent the user actually uses.
        """
        from framesleuth.mcp_server.server import render_fix_prompt

        bundle = await _load_bundle(job_id)
        return m.FixPromptResponse(
            id=job_id, action=bundle.get("action"), prompt=render_fix_prompt(bundle)
        )

    @app.get("/v1/report/{job_id}/timeline", tags=["reports"], responses={404: m.ERROR_404})
    async def report_timeline(job_id: str) -> Any:
        """Return the fused, cited event timeline written beside the bundle."""
        return await _read_sidecar_artifact(job_id, "timeline.json")

    @app.get("/v1/report/{job_id}/transcript", tags=["reports"], responses={404: m.ERROR_404})
    async def report_transcript(job_id: str) -> Any:
        """Return the (redacted) audio transcript written beside the bundle."""
        return await _read_sidecar_artifact(job_id, "transcript.json")

    async def _read_sidecar_artifact(job_id: str, filename: str) -> Any:
        """Read one JSON artifact from a bundle directory, with traversal checks."""
        job = await state.store.get_job(job_id)
        if job is None or not job.bundle_path:
            raise JobNotFoundError(job_id)
        path = Path(job.bundle_path).parent / filename
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail={"error": f"{filename} not available", "code": "missing_artifact"},
            )
        raw = await asyncio.to_thread(path.read_text, "utf-8")
        return json.loads(raw)

    @app.get(
        "/v1/report/{job_id}/keyframes/{index}",
        tags=["reports"],
        responses={404: m.ERROR_404},
        response_class=FileResponse,
    )
    async def report_keyframe(job_id: str, index: int) -> FileResponse:
        """Serve one analyzed keyframe image by its index in ``keyframe_refs``.

        The bundle cites ``frame:N`` throughout; without this endpoint an HTTP
        client could read those citations but never show the frame behind them.
        """
        job = await state.store.get_job(job_id)
        if job is None or not job.bundle_path:
            raise JobNotFoundError(job_id)
        bundle_dir = Path(job.bundle_path).parent.resolve()
        # `index` is client-supplied: resolve and confirm containment rather than
        # trusting it to be a plain integer path segment.
        target = (bundle_dir / "keyframes" / f"{index:03d}.png").resolve()
        if bundle_dir not in target.parents or not target.exists():
            raise HTTPException(
                status_code=404,
                detail={"error": f"Keyframe {index} not found", "code": "missing_keyframe"},
            )
        return FileResponse(target, media_type="image/png")

    @app.delete(
        "/v1/report/{job_id}",
        tags=["reports"],
        response_model=m.DeleteResponse,
        responses={404: m.ERROR_404, 409: m.ERROR_409},
    )
    async def delete_report(job_id: str) -> m.DeleteResponse:
        """Delete a finished report: its bundle directory and the job row.

        On-demand purge so a user isn't forced to wait for the TTL sweep. Only
        terminal jobs can be deleted — an active job must be cancelled first
        (``DELETE /v1/jobs/{id}``) so its running pipeline doesn't race the
        removal of the directory it is still writing to.
        """
        job = await state.store.get_job(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        if job.state.value not in _TERMINAL_STATES:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "Cancel the job before deleting its report",
                    "code": "job_active",
                },
            )
        # The bundle id is a UUID we minted (never client-controlled), but resolve
        # under the root and confirm containment before removing anything.
        bundle_root = settings.BUNDLE_DIR.resolve()
        target = (bundle_root / job.id).resolve()
        if target != bundle_root and bundle_root in target.parents:
            await asyncio.to_thread(shutil.rmtree, target, ignore_errors=True)
        await state.store.delete_job(job.id)
        return m.DeleteResponse(id=job.id, deleted=True)

    @app.get(
        "/v1/video/{job_id}",
        tags=["reports"],
        responses={404: m.ERROR_404},
        response_class=FileResponse,
    )
    async def get_video(job_id: str) -> FileResponse:
        """Serve the stored source recording with its correct media type."""
        job = await state.store.get_job(job_id)
        if job is None or not job.bundle_path:
            raise JobNotFoundError(job_id)
        bundle_dir = Path(job.bundle_path).parent
        source = _find_source_video(bundle_dir)
        if source is None:
            raise HTTPException(
                status_code=404, detail={"error": "Video missing", "code": "missing_video"}
            )
        # Serve the correct content-type per container; the source may be any
        # supported format (mp4/mov/mkv/avi), not just webm.
        media_type = _VIDEO_MEDIA_TYPES.get(source.suffix.lower(), "application/octet-stream")
        return FileResponse(source, media_type=media_type)

    @app.get(
        "/v1/gif/{job_id}",
        tags=["reports"],
        responses={404: m.ERROR_404, 422: m.ERROR_422},
        response_class=FileResponse,
    )
    async def get_gif(
        job_id: str,
        fps: float | None = Query(default=None, gt=0),
        width: float | None = Query(default=None, gt=0),
        start: float = Query(default=0.0, ge=0),
        end: float | None = Query(default=None, gt=0),
    ) -> FileResponse:
        """Render (and cache) an animated GIF preview of the recording.

        The client embeds this looping preview wherever a video player is awkward
        (issues, chat, the extension popup). Query params ``fps``/``width``/
        ``start``/``end`` are optional and clamped to safe ranges; results are
        cached on disk per parameter set, so repeat requests are served instantly.
        """
        from framesleuth.pipeline.gif import encode_gif, normalize_options

        job = await state.store.get_job(job_id)
        if job is None or not job.bundle_path:
            raise JobNotFoundError(job_id)
        bundle_dir = Path(job.bundle_path).parent
        source = _find_source_video(bundle_dir)
        if source is None:
            raise HTTPException(
                status_code=404, detail={"error": "Video missing", "code": "missing_video"}
            )

        options = normalize_options(
            fps=fps if fps is not None else settings.GIF_FPS,
            width=width if width is not None else settings.GIF_WIDTH,
            start=start,
            end=end,
            max_duration_s=settings.GIF_MAX_DURATION_S,
        )
        gif_path = bundle_dir / f"preview-{options.cache_key()}.gif"
        if not gif_path.exists():
            # PyAV decode + palette encode is CPU-bound — keep it off the loop.
            result = await asyncio.to_thread(encode_gif, source, gif_path, options=options)
            if result is None:
                raise HTTPException(
                    status_code=422,
                    detail={"error": "GIF encoding failed", "code": "gif_failed"},
                )
        return FileResponse(gif_path, media_type="image/gif")

    @app.post(
        "/v1/render-html",
        tags=["render"],
        responses={
            400: m.ERROR_422,
            413: m.ERROR_413,
            503: m.ERROR_503,
        },
        response_class=FileResponse,
    )
    async def render_html_endpoint(payload: m.RenderHtmlRequest) -> FileResponse:
        """Render an HTML document (CSS / JS / canvas animation) to a clip.

        Synchronous — returns the encoded file. Requires the optional ``render``
        extra (Playwright) and ``ffmpeg``; returns ``503`` with an actionable
        message when unavailable.
        """
        from framesleuth.pipeline.html_render import HtmlRenderError, RenderOptions, render_html

        html = payload.html
        if not html.strip():
            raise HTTPException(
                status_code=400, detail={"error": "Missing 'html'", "code": "missing_html"}
            )
        if len(html.encode("utf-8")) > settings.RENDER_MAX_HTML_BYTES:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": f"HTML exceeds {settings.RENDER_MAX_HTML_BYTES} bytes",
                    "code": "html_too_large",
                },
            )
        try:
            options = RenderOptions.normalized(
                fmt=payload.format,
                duration_s=payload.duration_s,
                fps=payload.fps,
                width=payload.width,
                height=payload.height,
            )
        except (HtmlRenderError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=400, detail={"error": str(exc), "code": "bad_options"}
            ) from exc

        out_dir = Path(tempfile.mkdtemp(prefix="fs-render-"))
        try:
            path = await render_html(
                html,
                options,
                out_dir,
                max_concurrency=settings.RENDER_MAX_CONCURRENCY,
                timeout_s=settings.RENDER_TIMEOUT_S,
            )
        except HtmlRenderError as exc:
            shutil.rmtree(out_dir, ignore_errors=True)
            raise HTTPException(
                status_code=503, detail={"error": str(exc), "code": "render_unavailable"}
            ) from exc
        except Exception:
            # Any other failure must not leak the fs-render- temp dir.
            shutil.rmtree(out_dir, ignore_errors=True)
            raise

        media = {"mp4": "video/mp4", "gif": "image/gif", "webm": "video/webm"}[options.fmt]
        return FileResponse(
            path,
            media_type=media,
            filename=f"animation.{options.fmt}",
            background=BackgroundTask(shutil.rmtree, out_dir, ignore_errors=True),
        )

    return app


app = create_app()


def main() -> None:  # pragma: no cover - process entrypoint
    """Run the HTTP backend on the configured BACKEND_HOST/BACKEND_PORT."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        # The lifespan drain owns waiting for in-flight jobs; give it room to run
        # instead of uvicorn's default, which would otherwise cut it short.
        timeout_graceful_shutdown=int(settings.SHUTDOWN_DRAIN_S) + 10,
        # `create_app` already installed the structured JSON handler on the
        # `framesleuth` logger; uvicorn's own dictConfig would reset logging and
        # undo it, so keep uvicorn out of the way.
        log_config=None,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
