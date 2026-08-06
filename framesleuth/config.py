"""Configuration management for Framesleuth.

Uses pydantic-settings for environment-based configuration with type safety.
All configuration is immutable after initialization.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment and .env file.

    All settings are typed, validated, and immutable. Follows the
    dependency inversion principle by centralizing configuration.
    """

    # === Server configuration ===
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8010  # 8000 left free for a capture backend, if you run one

    # === Model serving strategy ===
    # Defaults match the documented quick-start stack: native Ollama with the
    # qwen2.5vl vision model (README / docker-compose / .env.example all agree).
    # Point VLM_URL at llama.cpp or vLLM to swap engines — config only.
    VLM_URL: str = "http://127.0.0.1:11434"
    VLM_MODEL: str = "qwen2.5vl"
    CODER_URL: str = "http://127.0.0.1:11434"
    CODER_MODEL: str = "qwen2.5-coder:7b"

    # === Model parameters ===
    VLM_TIMEOUT_S: float = 60.0
    VLM_MAX_RETRIES: int = 3

    # Per-frame generation budget. The frame-analysis JSON is small, so a tight
    # cap keeps decode latency low; the focused error retry is allowed more room
    # for long stack traces (see ``VLM_ERROR_MAX_TOKENS``).
    VLM_MAX_TOKENS: int = 768
    VLM_ERROR_MAX_TOKENS: int = 1024
    # How many keyframes to send to the VLM at once. Local single-GPU servers
    # serialize internally, so keep this modest (raise only when the engine is
    # configured for parallelism, e.g. Ollama OLLAMA_NUM_PARALLEL / llama.cpp --parallel).
    VLM_MAX_CONCURRENCY: int = 3
    # Request OpenAI-style JSON output (``response_format``). Supported by
    # llama.cpp, Ollama, and vLLM; disable for an engine that rejects the field.
    VLM_JSON_MODE: bool = True
    # Ask for SCHEMA-guided decoding rather than plain JSON mode. `json_object`
    # only constrains "is it JSON", so a model can return valid JSON with none of
    # the fields the pipeline needs; a json_schema constrains the shape. All three
    # engines in the quick-start docs support it, but a proxy may not — hence a
    # flag rather than an unconditional upgrade.
    VLM_JSON_SCHEMA_MODE: bool = False
    # Transcode frames to JPEG before sending to the VLM to cut upload bytes and
    # vision prefill tokens. Stored keyframes stay lossless PNG regardless.
    VLM_SEND_JPEG: bool = True
    VLM_JPEG_QUALITY: int = 85

    # === Summary synthesis ===
    # Text model that turns the fused video+audio timeline into the summary.
    # Leave URL/MODEL blank to reuse the VLM endpoint (a vision-language model
    # summarizes text fine), so no extra model server is required.
    SUMMARY_URL: str = ""
    SUMMARY_MODEL: str = ""
    SUMMARY_TIMEOUT_S: float = 120.0
    SUMMARY_MAX_TOKENS: int = 1024

    # === Classification and routing ===
    CLASSIFY_CONFIDENCE_THRESHOLD: float = 0.7
    # When the deterministic heuristic lands in the ambiguous band
    # ``[floor, threshold)`` — some signal, but not confident — the orchestrator
    # may (a) resample extra frames around the suspected failure and (b) ask the
    # model to break the tie. Below the floor there is no signal worth the call.
    CLASSIFY_AMBIGUOUS_FLOOR: float = 0.3
    # Break ambiguous-band ties with a model classification (uses the summary
    # endpoint). Deterministic heuristics still decide the confident cases.
    CLASSIFY_USE_MODEL: bool = True
    # Bounded agentic resample: on an ambiguous classification with visual
    # evidence, re-sample extra frames around error timestamps and re-classify,
    # at most this many times. 0 disables it (pure single-pass).
    MAX_RESAMPLE_RETRIES: int = 2

    # === Transcription ===
    # Whisper hallucinates short filler phrases ("Thank you.", "Bye.") on silent
    # or near-silent audio with a high no-speech probability. Drop any segment
    # below this confidence so the timeline is not polluted with phantom speech.
    ASR_MIN_CONFIDENCE: float = 0.5
    # Run faster-whisper's built-in (Silero) voice-activity filter so silence is
    # never fed to the decoder — the most effective guard against hallucinated
    # speech, on top of the confidence floor above.
    ASR_VAD_FILTER: bool = True
    # Force a transcription language (ISO code, e.g. "en"); blank = auto-detect.
    ASR_LANGUAGE: str = ""
    # Whisper model size and compute type. Previously hard-coded to "small"/int8
    # with no way to trade accuracy for speed (or vice versa) on a given machine.
    # The model is cached per (name, compute_type) for the process lifetime.
    ASR_MODEL: str = "small"
    ASR_COMPUTE_TYPE: str = "int8"

    # === Upload and processing limits ===
    MAX_UPLOAD_MB: int = 512
    MAX_DURATION_S: int = 600  # 10 minutes
    MAX_FRAMES_PER_MIN: int = 30

    # === Frame processing ===
    # Frames are extracted/captioned at the low-res height to bound decode + VLM
    # cost. A suspected error frame with sparse OCR is re-decoded at the high-res
    # height (uncompressed) for a focused re-read, so tiny stack-trace text that a
    # 480p downscale would smear stays legible — "resolution where text lives".
    FRAME_LOWRES_HEIGHT: int = 480
    FRAME_HIGHRES_HEIGHT: int = 1080
    # Visual-delta score above which a frame is treated as a scene cut (fallback
    # keyframe selector). Higher = fewer, sharper cuts; lower = more sensitive.
    SCENE_CUT_THRESHOLD: float = 0.35
    # Collapse near-identical keyframes (held spinners, static title cards, repeated
    # screens) with a perceptual hash before the VLM, so the frame budget is spent
    # on distinct content. Conservative by default: only frames within
    # KEYFRAME_PHASH_HAMMING_MAX bits (out of 64) are treated as duplicates, so a
    # subtle but real on-screen change (e.g. an error appearing) is never dropped.
    KEYFRAME_DEDUP: bool = True
    KEYFRAME_PHASH_HAMMING_MAX: int = 4

    # Draw a marker on a keyframe when a click/cursor sidecar event with numeric
    # coordinates lands near it, so the VLM (and a human reviewer) can see exactly
    # where the user interacted. A no-op when events carry no coordinates.
    OVERLAY_INTERACTIONS: bool = True
    # Re-read suspected error frames with a dedicated OCR engine (pytesseract, the
    # optional ``[ocr]`` extra) when the VLM's OCR comes back sparse — a deterministic
    # backstop for tiny stack-trace text. A no-op when the extra is not installed.
    OCR_BACKSTOP: bool = True

    # === Grounding ===
    # Bound the workspace scan so a large monorepo cannot make grounding unbounded;
    # files beyond this count are skipped (and the truncation is logged).
    GROUNDING_MAX_FILES: int = Field(default=5000, ge=1)
    # Directories the HTTP API may ground against, comma-separated. Grounding
    # reads every source file under the given root and returns matching lines in
    # the bundle, so an unconstrained `repo_root` form field would let any
    # allowlisted browser origin read arbitrary files off this machine. Empty
    # (the default) disables `repo_root` over HTTP entirely; the MCP path is
    # unaffected, since there the caller is the local editor.
    GROUNDING_ROOTS: str = ""
    # Cap how many distinct search terms one analysis may derive from the video.
    # Evidence text is attacker-influenceable (it is whatever was on screen), and
    # the scan is O(files x lines x queries) — an unbounded query set turns a
    # stack-trace-heavy recording into a multi-minute CPU burn.
    GROUNDING_MAX_QUERIES: int = Field(default=64, ge=1, le=512)

    # === GIF preview ===
    # On-demand animated GIF rendered from the stored recording so a client can
    # embed a looping preview (issue, chat, extension) without a video player.
    # The window is capped at GIF_MAX_DURATION_S to keep the file small.
    GIF_FPS: int = 10
    GIF_WIDTH: int = 640
    GIF_MAX_DURATION_S: float = 30.0

    # === HTML → video rendering ===
    # Bound the optional Playwright render path: how many renders may run at
    # once, the total wall-clock budget per render (capture + encode), and the
    # largest accepted HTML document. Frame-by-frame capture materializes one
    # PNG per frame, so these keep a burst of large requests from exhausting
    # CPU/disk.
    RENDER_MAX_CONCURRENCY: int = 1
    RENDER_TIMEOUT_S: float = 300.0
    RENDER_MAX_HTML_BYTES: int = 2 * 1024 * 1024

    # === Storage ===
    BUNDLE_DIR: Path = Path("./bug-reports")
    DATABASE_PATH: Path = Path("./bug-reports/jobs.db")
    # Delete persisted bundles (and their job rows) older than this many days,
    # so disk use stays bounded. Swept at startup and then every
    # RETENTION_SWEEP_INTERVAL_S while the server runs. 0 disables cleanup.
    BUNDLE_TTL_DAYS: int = 0
    RETENTION_SWEEP_INTERVAL_S: float = 3600.0

    # === Job queue ===
    MAX_CONCURRENT_JOBS: int = Field(default=2, ge=1, le=64)
    JOB_TIMEOUT_S: int = Field(default=1800, ge=1)  # 30 minutes
    # MAX_CONCURRENT_JOBS bounds how many jobs *run*; this bounds how many may be
    # waiting. Without it a burst of uploads accumulates rows and multi-hundred-MB
    # temp files far faster than the pipeline drains them.
    MAX_QUEUED_JOBS: int = Field(default=32, ge=1)
    # Attempts per job, including the first. A local model server that is still
    # loading refuses connections for the first few seconds; retrying that costs
    # nothing, while re-uploading a 500 MB recording costs the user everything.
    # Only transient failures retry — bad input fails on attempt 1.
    JOB_MAX_ATTEMPTS: int = Field(default=2, ge=1, le=5)
    # How long shutdown waits for in-flight analyses before cancelling them and
    # recording a terminal state.
    SHUTDOWN_DRAIN_S: float = Field(default=30.0, ge=0.0)

    # === Notifications ===
    # POST a compact completion payload (job id, state, title, action) to this URL
    # when a job finishes, so an external system can react without polling. Blank
    # disables it. Loopback/private targets are allowed (it is your own webhook).
    WEBHOOK_URL: str = ""
    WEBHOOK_TIMEOUT_S: float = 10.0

    # === Logging ===
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_JSON: bool = True

    # === Security ===
    # Chrome extension IDs (comma-separated, 32 chars of a-p each) allowed to
    # call this API. Each becomes an exact `chrome-extension://<id>` origin in
    # the CORS allowlist. Blank = no extension origins. Never wildcarded: every
    # extension ID is 32 chars of [a-p], so a pattern like
    # `chrome-extension://[a-p]{32}` would allowlist *every* installed extension.
    CHROME_EXTENSION_IDS: str = ""
    # Browser origins (besides the extension) allowed to call this API directly —
    # e.g. the hosted site connecting to your locally-running agent. The deployed
    # marketing site + local dev are allowed by default so the "Try it" widget
    # works with zero config; override with a comma-separated WEB_ORIGINS.
    WEB_ORIGINS: str = (
        "https://framesleuth.com,https://www.framesleuth.com,"
        "http://localhost:3000,http://127.0.0.1:3000"
    )
    # Optional bearer token: when set, every /v1 endpoint except /v1/healthz
    # requires `Authorization: Bearer <API_TOKEN>`. CORS only controls what a
    # browser may *read* — it does not stop DNS-rebinding pages or other local
    # processes from *sending* requests. Set a token whenever that matters.
    API_TOKEN: str = ""
    # Also redact PII (emails, Luhn-valid card numbers, US SSNs/phones, cloud keys)
    # from OCR/console text, on top of the always-on secret/token patterns.
    REDACT_PII: bool = True
    # Text redaction scrubs the bundle's *text*, but it cannot scrub *pixels*: the
    # frame that showed a secret still contains it. When this is on and a run
    # actually redacted something, the source video (and, if
    # WITHHOLD_KEYFRAMES_ON_REDACTION is also on, the keyframe images) are NOT
    # persisted beside the bundle — closing the "bundle says [REDACTED] but ships
    # the secret as an image" gap. Off by default so replay/GIF keep working for
    # the common case; turn on for a security-conscious deployment.
    WITHHOLD_MEDIA_ON_REDACTION: bool = False
    WITHHOLD_KEYFRAMES_ON_REDACTION: bool = False

    @property
    def web_origins_list(self) -> list[str]:
        """Parsed, de-blanked list of allowed browser origins."""
        return [o.strip() for o in self.WEB_ORIGINS.split(",") if o.strip()]

    @property
    def grounding_roots_list(self) -> list[Path]:
        """Resolved directories the HTTP API may ground against (may be empty)."""
        roots: list[Path] = []
        for raw in self.GROUNDING_ROOTS.split(","):
            entry = raw.strip()
            if not entry:
                continue
            try:
                resolved = Path(entry).expanduser().resolve()
            except OSError:
                continue
            if resolved.is_dir():
                roots.append(resolved)
        return roots

    @property
    def chrome_extension_origins(self) -> list[str]:
        """Exact `chrome-extension://` origins built from ``CHROME_EXTENSION_IDS``."""
        return [
            f"chrome-extension://{ext_id.strip()}"
            for ext_id in self.CHROME_EXTENSION_IDS.split(",")
            if ext_id.strip()
        ]

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        # Ignore unknown keys so a .env written for an older release (with a
        # since-removed setting) can never stop the server from booting.
        "extra": "ignore",
    }

    def validate_paths(self) -> None:
        """Ensure required directories exist."""
        self.BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
        self.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the global settings instance (singleton pattern).

    Cached, as the docstring always claimed: without ``lru_cache`` every call
    re-parsed the environment *and* re-read ``.env`` from disk, and this is
    called from request-time paths (``preprocess``, ``classify``) that do not
    receive a settings object. Tests that mutate the environment should call
    ``get_settings.cache_clear()``.

    Returns:
        Settings: Validated application configuration.
    """
    return Settings()
