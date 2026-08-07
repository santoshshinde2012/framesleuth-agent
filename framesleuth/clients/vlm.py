"""OpenAI-compatible VLM client with retry, backoff, and structured output.

Follows the interface segregation principle: single focused responsibility.
Engine-agnostic HTTP abstraction per ADR-001.
"""

import asyncio
import base64
import inspect
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
from pydantic import BaseModel, Field, ValidationError, field_validator

from framesleuth.clients.model_catalog import looks_like_missing_model
from framesleuth.errors import ModelNotLoadedError, ModelUnavailableError
from framesleuth.logging_config import get_logger
from framesleuth.schemas import UiElement

if TYPE_CHECKING:  # pragma: no cover - typing only
    from framesleuth.config import Settings

logger = get_logger("clients.vlm")

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _backoff(attempt: int) -> float:
    """Exponential backoff with full jitter.

    Frames are analyzed concurrently, so a bare ``2 ** attempt`` had every
    in-flight request sleep for exactly the same interval and then retry in
    lockstep — re-creating the burst that overloaded the local model server in
    the first place. Jitter spreads them out.
    """
    return random.uniform(0.0, float(2**attempt))


async def _resolve_response(request: Any) -> Any:
    """Resolve aiohttp request object or coroutine into a response object.

    Supports both real aiohttp request context managers and AsyncMock-based
    test doubles that return awaitable responses directly.
    """
    if inspect.isawaitable(request):
        return await request
    return request


def _encode_image(image_path: str, *, send_jpeg: bool, jpeg_quality: int) -> tuple[str, str]:
    """Read an image and return ``(base64, mime_type)`` for an image_url payload.

    Transcodes to JPEG (smaller upload + fewer vision prefill tokens) when
    ``send_jpeg`` is set and OpenCV can decode the frame; otherwise the original
    bytes are sent unchanged. Never raises on a bad/opaque file — it falls back
    to the raw bytes so a single odd frame can't abort analysis.
    """
    raw = Path(image_path).read_bytes()
    if send_jpeg:
        try:
            import cv2  # lazy: heavy media dep, only in the full stack
            import numpy as np

            arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if arr is not None:
                ok, buf = cv2.imencode(
                    ".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
                )
                if ok:
                    return base64.b64encode(buf.tobytes()).decode("utf-8"), "image/jpeg"
        except Exception as exc:  # pragma: no cover - exercised only without cv2/bad image
            logger.debug("JPEG transcode skipped for %s: %s", image_path, exc)
    mime = (
        "image/png"
        if raw[:8] == _PNG_MAGIC or image_path.lower().endswith(".png")
        else "image/jpeg"
    )
    return base64.b64encode(raw).decode("utf-8"), mime


class FrameAnalysisResponse(BaseModel):
    """Response from VLM frame analysis.

    The build-aware fields (``screen_name`` … ``ui_elements``) are populated only by
    the build prompt for feature/demo videos; they default empty so the bug path is
    unaffected. ``ui_elements`` is parsed leniently — a malformed element is dropped
    rather than failing the whole frame.
    """

    # Both default to "" so a PARTIAL reply still yields a usable scene. When
    # they were required, a model that returned a perfect caption but omitted
    # `ocr_text` failed validation and the frame was discarded outright —
    # throwing away the caption too. Local models truncate JSON routinely, so
    # this was a steady, invisible drain on evidence.
    caption: str = ""
    ocr_text: str = ""
    ui_action: str | None = None
    is_error_state: bool = False
    reason: str | None = None
    screen_name: str | None = None
    layout: str | None = None
    design_notes: str | None = None
    data_shown: str | None = None
    ui_elements: list[UiElement] = Field(default_factory=list)

    @field_validator("ui_elements", mode="before")
    @classmethod
    def _coerce_ui_elements(cls, v: Any) -> list[dict[str, Any]]:
        """Keep only well-formed elements (dict with kind+label); cap the count."""
        if not isinstance(v, list):
            return []
        out: list[dict[str, Any]] = []
        for item in v:
            if isinstance(item, dict) and item.get("kind") and item.get("label"):
                out.append(item)
            if len(out) >= 24:  # bound runaway model output
                break
        return out


class VLMClient:
    """OpenAI-compatible VLM (Qwen3-VL) client.

    Handles:
    - Retry with exponential backoff
    - JSON parsing with fallback
    - Timeout and circuit-breaker guards
    - Structured output (JSON schema per engine)
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        *,
        max_tokens: int = 768,
        json_mode: bool = True,
        send_jpeg: bool = True,
        jpeg_quality: int = 85,
        breaker_threshold: int = 3,
        json_schema_mode: bool = False,
    ) -> None:
        """Initialize VLM client.

        Args:
            base_url: OpenAI-compatible API base URL (e.g., http://127.0.0.1:8080).
            model: Model identifier (e.g., Qwen/Qwen3-VL-8B-Instruct-GGUF).
            timeout_s: Request timeout in seconds.
            max_retries: Maximum number of retries on 5xx/timeout.
            max_tokens: Default generation cap per frame.
            json_mode: Request OpenAI-style ``response_format`` JSON output.
            send_jpeg: Transcode frames to JPEG before upload.
            jpeg_quality: JPEG quality (1-100) when ``send_jpeg`` is set.
            breaker_threshold: Consecutive failures after which the client stops
                trying and fails fast, so one dead server does not cost every
                frame its full retry budget.
            json_schema_mode: Request schema-guided decoding (constrains which
                KEYS appear, not just that the reply is JSON). Off by default
                because a proxy in front of the engine may reject the field.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.send_jpeg = send_jpeg
        self.jpeg_quality = jpeg_quality
        self.breaker_threshold = max(1, breaker_threshold)
        self.json_schema_mode = json_schema_mode
        self._session: aiohttp.ClientSession | None = None
        # Circuit breaker state. Each frame previously retried independently, so
        # a hard-down model server cost (frames / concurrency) x max_retries x
        # timeout — roughly 12 minutes for a 12-frame job — before the stage
        # finally degraded. After `breaker_threshold` consecutive failures we
        # stop trying and let the stage degrade in seconds instead.
        self._consecutive_failures = 0
        self._breaker_open = False

    @property
    def breaker_open(self) -> bool:
        """Whether the breaker has tripped (the server is considered down)."""
        return self._breaker_open

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._breaker_open = False

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.breaker_threshold and not self._breaker_open:
            self._breaker_open = True
            logger.error(
                "VLM circuit breaker opened after %d consecutive failures; "
                "remaining frames will fail fast",
                self._consecutive_failures,
            )

    @classmethod
    def from_settings(cls, settings: "Settings") -> "VLMClient":
        """Build a client from app settings.

        The single construction path for both the HTTP service and the MCP
        server, so VLM tuning (timeouts, token caps, JSON mode, JPEG) stays
        consistent wherever a frame is analyzed.
        """
        return cls(
            settings.VLM_URL,
            settings.VLM_MODEL,
            timeout_s=settings.VLM_TIMEOUT_S,
            max_retries=settings.VLM_MAX_RETRIES,
            max_tokens=settings.VLM_MAX_TOKENS,
            json_mode=settings.VLM_JSON_MODE,
            send_jpeg=settings.VLM_SEND_JPEG,
            jpeg_quality=settings.VLM_JPEG_QUALITY,
            json_schema_mode=settings.VLM_JSON_SCHEMA_MODE,
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a shared, reused HTTP session (created lazily).

        Reusing one session keeps the TCP/TLS connection pool warm across the
        many per-frame calls instead of paying connection setup on each request.
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def aclose(self) -> None:
        """Close the shared HTTP session, if open."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> "VLMClient":
        """Enter an async context that closes the session on exit."""
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Close the shared session when leaving the context."""
        await self.aclose()

    async def analyze_frame(
        self,
        image_path: str,
        timestamp: float,
        prompt_override: str | None = None,
        *,
        max_tokens: int | None = None,
        send_jpeg: bool | None = None,
    ) -> FrameAnalysisResponse:
        """Analyze a single frame using the VLM.

        Args:
            image_path: Path to the frame image.
            timestamp: Timestamp of the frame in seconds (for context).
            prompt_override: Custom prompt; if None, uses default.
            max_tokens: Per-call generation cap (defaults to ``self.max_tokens``).
            send_jpeg: Per-call JPEG override (defaults to ``self.send_jpeg``). Pass
                ``False`` to send the frame uncompressed — used for error re-OCR so
                tiny error text is never degraded by lossy compression.

        Returns:
            Parsed frame analysis response.

        Raises:
            ModelUnavailableError: If VLM is unreachable or fails.
            ValidationError: If response doesn't match schema.
        """
        from framesleuth.prompts import VLMPrompts

        if prompt_override is None:
            prompt = VLMPrompts.frame_analysis(timestamp)
        else:
            prompt = prompt_override

        use_jpeg = self.send_jpeg if send_jpeg is None else send_jpeg
        image_data, mime = _encode_image(
            image_path, send_jpeg=use_jpeg, jpeg_quality=self.jpeg_quality
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_data}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.0,  # Deterministic OCR/structured output.
            "max_tokens": max_tokens or self.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = self._response_format()

        response = await self._call_with_retry(payload)
        try:
            return self._parse_response(response)
        except ValueError:
            # A malformed BODY is not a transport failure, so `_call_with_retry`
            # never retried it — the frame was simply dropped. Local models
            # truncate JSON routinely (understand.py calls it "routine"), so one
            # bounded re-ask with more room recovers a real share of them.
            logger.warning("VLM returned unparseable JSON; re-asking once for this frame")
            repair = dict(payload)
            repair["max_tokens"] = int((max_tokens or self.max_tokens) * 1.5)
            repair["messages"] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_data}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                f"{prompt}\n\nIMPORTANT: your previous reply was not valid JSON. "
                                "Return ONLY a single complete JSON object and nothing else."
                            ),
                        },
                    ],
                }
            ]
            return self._parse_response(await self._call_with_retry(repair))

    def _response_format(self) -> dict[str, Any]:
        """The OpenAI-style ``response_format`` to request.

        ``{"type": "json_object"}`` constrains only "is it JSON" — not which keys
        appear — so a model can return valid JSON with none of the fields we
        need. All three engines named in the config (llama.cpp, Ollama, vLLM)
        support schema-guided decoding, which constrains the *shape*.

        Falls back to plain JSON mode when ``VLM_JSON_SCHEMA_MODE`` is off, since
        a proxy in front of the engine may reject the richer form.
        """
        if not self.json_schema_mode:
            return {"type": "json_object"}
        schema = FrameAnalysisResponse.model_json_schema()
        return {
            "type": "json_schema",
            "json_schema": {"name": "frame_analysis", "schema": schema, "strict": False},
        }

    async def _call_with_retry(self, payload: dict[str, Any]) -> str:  # noqa: C901
        """Call the VLM with exponential backoff retry.

        Args:
            payload: Request payload.

        Returns:
            Raw response text.

        Raises:
            ModelUnavailableError: If all retries exhausted.
        """
        url = f"{self.base_url}/v1/chat/completions"

        if self._breaker_open:
            # Already established that the server is down; don't spend another
            # `max_retries * timeout_s` proving it for every remaining frame.
            raise ModelUnavailableError("VLM", self.base_url)

        for attempt in range(self.max_retries):
            try:
                session = await self._get_session()
                resolved = await _resolve_response(session.post(url, json=payload))
                if hasattr(resolved, "__aenter__"):
                    async with resolved as resp:
                        text = await self._handle_status(resp, attempt)
                else:
                    text = await self._handle_status(resolved, attempt)
                if text is not None:
                    self._record_success()
                    return text
                await asyncio.sleep(_backoff(attempt))
            except TimeoutError:
                logger.warning(f"VLM timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(_backoff(attempt))
                    continue
                self._record_failure()
                raise ModelUnavailableError("VLM", self.base_url) from None
            except ModelUnavailableError:
                self._record_failure()
                raise
            except (aiohttp.ClientError, ConnectionError, OSError) as e:
                # Connection refused/reset is the *most common* transient failure
                # with a local model server that is (re)loading a model — retry it
                # with the same backoff as a 5xx instead of failing on attempt 1.
                logger.warning(
                    f"VLM connection error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(_backoff(attempt))
                    continue
                self._record_failure()
                raise ModelUnavailableError("VLM", self.base_url) from e
            except Exception as e:
                logger.error(f"VLM request failed: {e}")
                self._record_failure()
                raise ModelUnavailableError("VLM", self.base_url) from e

        self._record_failure()
        raise ModelUnavailableError("VLM", self.base_url)

    async def _handle_status(self, resp: Any, attempt: int) -> str | None:
        """Map an HTTP response to text, a retry signal, or a fatal error.

        Args:
            resp: Response object exposing ``status`` and ``text()``.
            attempt: Zero-based attempt index for retry accounting.

        Returns:
            Response text on success, or ``None`` to signal a retriable 503.

        Raises:
            ModelNotLoadedError: When the server answers but the model is absent.
            ModelUnavailableError: On other non-retriable or exhausted failures.
        """
        if resp.status == 200:
            text: str = await resp.text()
            logger.info(f"VLM call successful (attempt {attempt + 1})")
            return text
        # Transient server-side failures (overload, gateway, rate limit) are worth
        # retrying; 4xx client errors (bad payload, model not found) are not — they
        # would fail identically on every attempt, so surface them immediately.
        retriable = resp.status in (429, 500, 502, 503, 504)
        if retriable and attempt < self.max_retries - 1:
            logger.warning(f"VLM returned {resp.status} (attempt {attempt + 1}/{self.max_retries})")
            return None
        if not retriable:
            body = (await resp.text())[:500]
            logger.error(f"VLM returned non-retriable {resp.status}: {body}")
            # A 404 (or an engine that phrases it as 400 "model not found") means
            # the server IS answering and only the model is missing. Collapsing
            # that into "not reachable at <url>" sent operators to restart a
            # server that was already up instead of pulling the model.
            # NOTE: the body must actually say so. A bare 404 is ambiguous —
            # a wrong base_url, a reverse proxy, or an engine not serving
            # /v1/chat/completions at that path all return one, and reporting
            # those as "model not loaded" would repeat the exact misattribution
            # this branch exists to remove, just pointing the other way.
            if looks_like_missing_model(body):
                raise ModelNotLoadedError("VLM", self.model, self.base_url)
        raise ModelUnavailableError("VLM", self.base_url)

    def _parse_response(self, response_text: str) -> FrameAnalysisResponse:
        """Parse and validate VLM response.

        Args:
            response_text: Raw response from VLM.

        Returns:
            Validated FrameAnalysisResponse.

        Raises:
            ValueError: If response cannot be parsed or doesn't match schema.
        """
        try:
            response_data = json.loads(response_text)
            # Extract JSON from choices[0].message.content
            if "choices" in response_data:
                content = response_data["choices"][0]["message"]["content"]
            else:
                content = response_text

            # Try to parse as JSON if it's a string
            if isinstance(content, str):
                # Extract JSON object from potential markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                content = json.loads(content)
            if not isinstance(content, dict):
                raise ValueError(f"VLM content is {type(content).__name__}, expected object")

            # Validate against schema
            parsed = FrameAnalysisResponse(**content)
            logger.debug(
                f"VLM response parsed: error_state={parsed.is_error_state}, "
                f"caption_len={len(parsed.caption)}"
            )
            return parsed

        # IndexError: empty ``choices``; TypeError: null/list content — both are
        # things a weak or proxying server actually returns, so they must map to
        # the documented ValueError, not escape as raw exceptions.
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValidationError) as e:
            logger.error(f"Failed to parse VLM response: {e}")
            raise ValueError(f"VLM response validation failed: {e}") from e

    async def warmup(self) -> bool:
        """Warm up the VLM by calling it with a dummy request.

        Returns:
            True if warmup successful, False otherwise.
        """
        import tempfile

        logger.info("Warming up VLM...")
        # A dummy 1x1 PNG written to a private temp file (never a shared, fixed
        # /tmp path that could collide or linger between runs).
        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            handle.write(png_data)
            dummy_frame = handle.name
        try:
            await self.analyze_frame(dummy_frame, 0.0)
            logger.info("VLM warmup complete")
            return True
        except Exception as e:
            logger.warning(f"VLM warmup failed: {e}")
            return False
        finally:
            Path(dummy_frame).unlink(missing_ok=True)
