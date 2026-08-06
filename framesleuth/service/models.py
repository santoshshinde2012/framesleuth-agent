"""Typed request/response contracts for the HTTP surface.

Every route declares one of these, so ``/openapi.json`` is a *complete* machine
description of the API rather than a wall of untyped ``object``s. That is what
makes a generated, always-in-sync client possible (see
``scripts/export_openapi.py`` and the website's generated types) — an untyped
schema silently drifts from the server and the drift is only found at runtime.

Kept separate from :mod:`framesleuth.schemas` (the *analysis* contract) because
these describe the *transport*: job envelopes, listings, and error shapes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from framesleuth.schemas import JobState

# ===== Errors =====


class ErrorBody(BaseModel):
    """The body of every non-2xx response.

    This is the WHOLE body, not a nested field. The API installs exception
    handlers that flatten FastAPI's default ``{"detail": {...}}`` wrapper, so a
    client reads ``code`` directly off the response.

    ``code`` is the stable, machine-readable discriminator (see
    :class:`framesleuth.errors.ErrorCode`); ``error`` is human text, ``hint`` an
    optional actionable next step, and ``request_id`` matches the
    ``X-Request-ID`` response header so one value ties a report to its logs.
    Clients should branch on ``code``.
    """

    error: str = Field(..., description="Human-readable description of what went wrong")
    code: str = Field(..., description="Stable machine-readable error code")
    hint: str | None = Field(None, description="Actionable suggestion for fixing it")
    request_id: str | None = Field(
        None, description="Correlation id, also returned as the X-Request-ID header"
    )


# Reusable OpenAPI ``responses`` fragments so every route documents its failures.
# These reference ErrorBody directly: an earlier `ErrorResponse` wrapper class
# described the pre-flattening `{"detail": ...}` shape, which meant the published
# schema disagreed with every actual error the server returns — and the
# schema-freshness CI check happily certified it as current.
ERROR_404 = {"model": ErrorBody, "description": "Job or report not found"}
ERROR_409 = {"model": ErrorBody, "description": "Job is not in a valid state for this call"}
ERROR_413 = {"model": ErrorBody, "description": "Payload exceeds the configured limit"}
ERROR_422 = {"model": ErrorBody, "description": "Input could not be processed"}
ERROR_503 = {"model": ErrorBody, "description": "An optional capability is unavailable"}


# ===== Meta =====


class VersionResponse(BaseModel):
    """Build and capability advertisement.

    A client (the website, an extension, a CI check) reads this *once* to learn
    which optional features this particular install actually has, instead of
    probing endpoints and interpreting 404/503 as "not supported".
    """

    name: str = "framesleuth"
    version: str = Field(..., description="Package version of the running agent")
    schema_version: str = Field(..., description="Context Bundle schema version emitted")
    api_version: str = Field(default="v1", description="HTTP API version prefix")
    capabilities: dict[str, bool] = Field(
        default_factory=dict,
        description="Optional capabilities in THIS install: render_html, ocr_backstop, "
        "auth_required, webhook, redact_pii, grounding_over_http",
    )
    render: RenderAvailability | None = Field(
        default=None,
        description="Full HTML->video diagnostics, including which interpreter the "
        "server runs and what is missing. Lives here rather than on /v1/healthz "
        "because this endpoint is covered by the API_TOKEN gate.",
    )


class ServiceHealthModel(BaseModel):
    """Readiness of one dependency."""

    name: str
    status: Literal["ready", "unavailable", "degraded"] | str
    latency_ms: float | None = None
    error: str | None = None


class RenderAvailability(BaseModel):
    """Whether the optional HTML→video path can run in this process."""

    playwright: bool = False
    playwright_version: str | None = None
    chromium: bool = False
    ffmpeg: bool = False
    python: str | None = None
    ready: bool = False
    hint: str | None = None


class HealthResponse(BaseModel):
    """``GET /v1/healthz`` — dependency readiness and live queue depth."""

    status: Literal["healthy", "degraded", "unhealthy"] | str
    services: dict[str, ServiceHealthModel]
    queue_depth: int = 0
    timestamp: str = ""
    render: RenderAvailability | None = None


class NamedDescription(BaseModel):
    """A catalog entry: a selectable name plus what it does."""

    name: str
    description: str


class SkillsResponse(BaseModel):
    """``GET /v1/skills`` — the summary styles ``analyze`` accepts."""

    default: str
    skills: list[NamedDescription]


class ActionsResponse(BaseModel):
    """``GET /v1/actions`` — the action modes ``analyze`` accepts."""

    default: str
    auto: bool = Field(True, description="Whether an omitted action is auto-picked")
    actions: list[NamedDescription]


# ===== Jobs =====


class AnalyzeAccepted(BaseModel):
    """``POST /v1/analyze`` — 202 handoff to the background pipeline."""

    job_id: str
    status: JobState
    idempotent: bool = Field(
        False, description="True when identical bytes+options reused an existing job"
    )


class JobSummary(BaseModel):
    """One row of the job listing."""

    id: str
    state: JobState
    progress_pct: int = Field(..., ge=0, le=100)
    source_video: str
    created_at: str
    updated_at: str


class JobListResponse(BaseModel):
    """``GET /v1/jobs`` — a page of history, newest first."""

    total: int
    limit: int
    offset: int
    jobs: list[JobSummary]


class JobDetail(BaseModel):
    """``GET /v1/jobs/{id}`` — lifecycle state plus live pipeline metrics."""

    id: str
    state: JobState
    progress_pct: int = Field(..., ge=0, le=100)
    source_video: str | None = None
    bundle_path: str | None = Field(
        None, description="Bundle path relative to the bundle root, once available"
    )
    error: ErrorBody | dict[str, Any] | None = None
    metrics: dict[str, Any] | None = Field(
        None, description="Per-stage timings and degraded stages, while/after the run"
    )
    cancel_requested: bool = False
    created_at: str
    updated_at: str


class JobCancelResponse(BaseModel):
    """``DELETE /v1/jobs/{id}`` — cooperative cancellation acknowledgement."""

    id: str
    state: JobState
    cancel_requested: bool = True


class DeleteResponse(BaseModel):
    """``DELETE /v1/report/{id}`` — the bundle directory and job row are gone."""

    id: str
    deleted: bool = True


class RenderedArtifact(BaseModel):
    """``GET /v1/report/{id}/render`` — a projection of the bundle as text."""

    id: str
    format: Literal["markdown", "issue", "test-plan"]
    content_type: str = Field(default="text/markdown", description="Suggested MIME type for `text`")
    text: str


class FixPromptResponse(BaseModel):
    """``GET /v1/report/{id}/prompt`` — the grounded, evidence-only action prompt."""

    id: str
    action: str | None = None
    prompt: str


class RenderHtmlRequest(BaseModel):
    """``POST /v1/render-html`` — an HTML animation to capture as a clip.

    Fields are intentionally *unconstrained* here. The renderer
    (``RenderOptions.normalized``) is the single authority on what is a valid
    combination, and it reports problems as the documented
    ``missing_html`` / ``bad_options`` / ``html_too_large`` codes. Duplicating
    those bounds as Pydantic constraints would turn every one of them into a
    generic 422 and split the contract across two places.
    """

    html: str = Field("", description="A self-contained HTML document (inline CSS/JS)")
    format: str = Field("mp4", description="mp4 | gif | webm")
    duration_s: float = Field(5.0, description="Clip length in seconds (max 30)")
    fps: int = Field(30, description="5-60")
    width: int = Field(1280, description="64-3840")
    height: int = Field(720, description="64-2160")
