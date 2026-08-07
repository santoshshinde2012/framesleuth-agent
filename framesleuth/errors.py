"""Typed exception taxonomy for Framesleuth.

Follows a clear exception hierarchy with user-friendly error codes and hints.
Enables the API layer to translate exceptions into appropriate HTTP responses.
"""

from enum import StrEnum


class ErrorCode(StrEnum):
    """Machine-readable error codes for client error handling."""

    # Video/input errors
    UNSUPPORTED_MEDIA = "unsupported_media"
    PROBE_FAILED = "probe_failed"
    NO_VIDEO_TRACK = "no_video_track"
    DURATION_EXCEEDED = "duration_exceeded"
    UPLOAD_TOO_LARGE = "upload_too_large"

    # Model availability
    VLM_UNAVAILABLE = "vlm_unavailable"
    CODER_UNAVAILABLE = "coder_unavailable"
    ASR_UNAVAILABLE = "asr_unavailable"
    # Distinct from the two above: the server answered, the *model* is missing.
    MODEL_NOT_LOADED = "model_not_loaded"

    # Processing errors
    PREPROCESSING_FAILED = "preprocessing_failed"
    TRANSCRIPT_GENERATION_FAILED = "transcript_generation_failed"
    FRAME_EXTRACTION_FAILED = "frame_extraction_failed"
    UNDERSTANDING_FAILED = "understanding_failed"

    # Job/state errors
    JOB_NOT_FOUND = "job_not_found"
    JOB_TIMEOUT = "job_timeout"
    JOB_CANCELLED = "job_cancelled"
    INVALID_STATE_TRANSITION = "invalid_state_transition"

    # Generic
    INTERNAL_ERROR = "internal_error"


class FramesleutheException(Exception):
    """Base exception for all Framesleuth errors.

    Provides structured error information for consistent error handling.
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        hint: str | None = None,
        status_code: int = 500,
    ) -> None:
        """Initialize exception with structured error info.

        Args:
            message: Human-readable error description.
            code: Machine-readable error code.
            hint: Optional actionable hint for users.
            status_code: HTTP status code to return.
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.hint = hint
        self.status_code = status_code

    def to_dict(self) -> dict[str, str]:
        """Convert exception to error response dict.

        Returns:
            Dictionary with error, code, and hint fields.
        """
        result: dict[str, str] = {
            "error": self.message,
            "code": self.code.value,
        }
        if self.hint:
            result["hint"] = self.hint
        return result


class UnsupportedMediaError(FramesleutheException):
    """Raised when video format is not supported."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        """Initialize with default code and status."""
        if hint is None:
            hint = "Try converting to H.264 MP4 or VP9 WebM using ffmpeg."
        super().__init__(
            message=message,
            code=ErrorCode.UNSUPPORTED_MEDIA,
            hint=hint,
            status_code=422,
        )


class ModelUnavailableError(FramesleutheException):
    """Raised when required model server is unreachable."""

    def __init__(self, model_name: str, url: str) -> None:
        """Initialize with model-specific details.

        Args:
            model_name: Name of the unavailable model (VLM, coder, etc.).
            url: URL where model server should be running.
        """
        message = f"Model server {model_name} not reachable at {url}"
        hint = f"Ensure {model_name} is running and accessible."
        super().__init__(
            message=message,
            code=ErrorCode.VLM_UNAVAILABLE if model_name == "VLM" else ErrorCode.CODER_UNAVAILABLE,
            hint=hint,
            status_code=503,
        )


class ModelNotLoadedError(ModelUnavailableError):
    """Raised when the model server answers but the requested model is absent.

    Split out from its parent because the *fix* is different: the server is up,
    so restarting it, checking the port, or re-reading the URL changes nothing —
    the operator has to pull/load the model (or point config at one that is
    loaded). Reporting this as "not reachable at <url>" sent people to debug a
    server that was already answering.

    Deliberately a *subclass* of ModelUnavailableError: every existing
    ``except ModelUnavailableError`` path (the job-retry accounting in the API,
    the 503 envelope, the clients' own re-raise guards) keeps catching it
    unchanged, and only the message/code get more specific.
    """

    def __init__(self, model_name: str, model: str, url: str) -> None:
        """Initialize with the server role, the missing model, and the URL.

        Args:
            model_name: Role of the server ("VLM", "Coder").
            model: The model identifier that was requested but is not loaded.
            url: URL of the (reachable) model server.
        """
        # Skips ModelUnavailableError.__init__, whose message hard-codes
        # "not reachable" — the exact wording this class exists to avoid.
        FramesleutheException.__init__(
            self,
            message=f"Model '{model}' is not loaded on the {model_name} server at {url}",
            code=ErrorCode.MODEL_NOT_LOADED,
            hint=(
                f"The server is reachable — the model is not. Load it "
                f"(e.g. `ollama pull {model}`) or set {model_name.upper()}_MODEL "
                f"to a model this server has."
            ),
            status_code=503,
        )
        self.model = model


class UploadTooLargeError(FramesleutheException):
    """Raised when upload exceeds size limit."""

    def __init__(self, size_mb: float, max_mb: int) -> None:
        """Initialize with size details.

        Args:
            size_mb: Actual file size in MB.
            max_mb: Maximum allowed size in MB.
        """
        message = f"Upload size {size_mb:.1f} MB exceeds limit of {max_mb} MB"
        hint = f"Record a shorter video (limit: {max_mb} MB)."
        super().__init__(
            message=message,
            code=ErrorCode.UPLOAD_TOO_LARGE,
            hint=hint,
            status_code=413,
        )


class DurationExceededError(FramesleutheException):
    """Raised when video duration exceeds limit."""

    def __init__(self, duration_s: float, max_s: int) -> None:
        """Initialize with duration details.

        Args:
            duration_s: Actual video duration in seconds.
            max_s: Maximum allowed duration in seconds.
        """
        message = f"Video duration {duration_s:.1f}s exceeds limit of {max_s}s"
        hint = f"Record a shorter video (limit: {max_s}s)."
        super().__init__(
            message=message,
            code=ErrorCode.DURATION_EXCEEDED,
            hint=hint,
            status_code=422,
        )


class LowEvidenceWarning(Exception):
    """Non-fatal warning when evidence for a field is weak.

    These are collected and recorded in the bundle but do not prevent
    generation of a partial result.
    """

    def __init__(self, field: str, reason: str) -> None:
        """Initialize with field and reason.

        Args:
            field: Name of the field with low evidence.
            reason: Explanation of why evidence is low.
        """
        super().__init__(f"Low evidence for {field}: {reason}")
        self.field = field
        self.reason = reason


class PreprocessingFailedError(FramesleutheException):
    """Raised when video preprocessing fails."""

    def __init__(self, reason: str) -> None:
        """Initialize with preprocessing failure reason.

        Args:
            reason: Specific reason for preprocessing failure.
        """
        super().__init__(
            message=f"Failed to preprocess video: {reason}",
            code=ErrorCode.PREPROCESSING_FAILED,
            hint="Check video format and integrity.",
            status_code=422,
        )


class JobNotFoundError(FramesleutheException):
    """Raised when a job is not found in the store."""

    def __init__(self, job_id: str) -> None:
        """Initialize with job ID.

        Args:
            job_id: The requested job ID.
        """
        super().__init__(
            message=f"Job {job_id} not found",
            code=ErrorCode.JOB_NOT_FOUND,
            hint="Check that the job ID is correct.",
            status_code=404,
        )


class JobCancelledError(FramesleutheException):
    """Raised inside the pipeline when a job has been cancelled by the caller."""

    def __init__(self, job_id: str) -> None:
        """Initialize with the cancelled job's ID."""
        super().__init__(
            message=f"Job {job_id} was cancelled",
            code=ErrorCode.JOB_CANCELLED,
            hint="The job was cancelled via DELETE /v1/jobs/{id}.",
            status_code=409,
        )


class JobTimeoutError(FramesleutheException):
    """Raised when a job exceeds its time limit."""

    def __init__(self, job_id: str, timeout_s: int) -> None:
        """Initialize with job details.

        Args:
            job_id: The timed-out job ID.
            timeout_s: The timeout limit in seconds.
        """
        super().__init__(
            message=f"Job {job_id} exceeded timeout of {timeout_s}s",
            code=ErrorCode.JOB_TIMEOUT,
            hint="Try a shorter recording or check system resources.",
            status_code=504,
        )


def sanitize_message(message: str) -> str:
    """Strip absolute filesystem paths from an error string before it is surfaced.

    Generic exceptions often embed the user's home directory or a temp path
    (``/Users/alice/...``, ``/tmp/upload-...``). Those responses are readable by
    allowlisted web origins, so the layout should not leak. Path-like tokens are
    replaced with their basename; the rest of the message is preserved.
    """
    import re

    # Replace any run of non-space chars that looks like an absolute POSIX path
    # (two or more segments) with just its final component.
    def _basename(match: "re.Match[str]") -> str:
        token = match.group(0)
        return token.rsplit("/", 1)[-1] or token

    return re.sub(r"/(?:[^/\s]+/)+[^/\s]*", _basename, message)
