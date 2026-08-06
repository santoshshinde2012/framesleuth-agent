"""Ingest browser-captured sidecars into structured analysis inputs.

The Chrome extension emits a resilient flat event stream (``SidecarEvent[]``)
where each event carries a ``source`` discriminator. This module normalizes
that stream (or the structured ``Sidecars`` object) into typed evidence,
reproduction steps, and environment metadata so a meaningful bundle can be
produced even when no vision model is available (graceful degradation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from framesleuth.logging_config import get_logger
from framesleuth.schemas import ErrorEvidenceItem, ReproStep

logger = get_logger("pipeline.sidecars")

# Network responses at or above this status are treated as failures.
_HTTP_ERROR_THRESHOLD = 400
# Cap on distinct error-evidence items kept, so an error loop (a React render
# loop logging the same exception 60x/s) can't inflate the bundle or the
# grounding query set into the thousands.
_MAX_EVIDENCE_ITEMS = 200


@dataclass
class ParsedSidecars:
    """Normalized view of browser sidecar events."""

    console_errors: list[dict[str, Any]] = field(default_factory=list)
    network: list[dict[str, Any]] = field(default_factory=list)
    clicks: list[dict[str, Any]] = field(default_factory=list)
    cursor: list[dict[str, Any]] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """True when no actionable signal was captured."""
        return not (self.console_errors or self.network or self.clicks)


def parse_sidecars(raw: Any) -> ParsedSidecars:
    """Parse sidecars from the flat event stream or the structured object.

    Accepts:
    - a flat ``list`` of events, each tagged with ``source`` (extension format),
    - a structured ``dict`` with ``console_errors``/``network``/``clicks``/``env``,
    - ``None`` or malformed input (returns an empty bundle, never raises).
    """
    if raw is None:
        return ParsedSidecars()

    if isinstance(raw, dict):
        return _parse_structured(raw)

    if isinstance(raw, list):
        return _parse_flat(raw)

    logger.warning("Unrecognized sidecar payload type: %s", type(raw).__name__)
    return ParsedSidecars()


def _parse_structured(raw: dict[str, Any]) -> ParsedSidecars:
    """Parse the documented structured ``Sidecars`` dict."""
    env_raw = raw.get("env")
    env: dict[str, Any] = env_raw if isinstance(env_raw, dict) else {}
    return ParsedSidecars(
        console_errors=[c for c in _as_list(raw.get("console_errors")) if isinstance(c, dict)],
        network=[n for n in _as_list(raw.get("network")) if isinstance(n, dict)],
        clicks=[c for c in _as_list(raw.get("clicks")) if isinstance(c, dict)],
        cursor=[c for c in _as_list(raw.get("cursor")) if isinstance(c, dict)],
        env=env,
    )


def _parse_flat(raw: list[Any]) -> ParsedSidecars:
    """Parse the extension's flat ``SidecarEvent[]`` stream by ``source`` tag."""
    parsed = ParsedSidecars()
    for event in raw:
        if not isinstance(event, dict):
            continue
        source = event.get("source")
        if source == "console":
            parsed.console_errors.append(event)
        elif source == "network":
            parsed.network.append(event)
        elif source == "click":
            parsed.clicks.append(event)
        elif source == "cursor":
            parsed.cursor.append(event)
        elif source == "env":
            # The latest env snapshot wins.
            parsed.env = {k: v for k, v in event.items() if k not in {"t", "source"}}
    return parsed


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _is_failed_request(entry: dict[str, Any]) -> bool:
    """Whether a captured network entry represents a failure.

    A 4xx/5xx status is a failure, and so is a request that never got a response
    — connection refused, CORS block, ``net::ERR_*`` — which the extension emits
    as ``status: 0`` or with an explicit ``error``/``failed`` marker. Treating
    only ``status >= 400`` as failure silently drops the classic "the API is
    down" bug, where every fetch fails with status 0.
    """
    status = _to_int(entry.get("status"))
    if status >= _HTTP_ERROR_THRESHOLD:
        return True
    if status <= 0:
        # No response at all. Distinguish a genuine failure from an entry we
        # simply have no status for by requiring some failure marker.
        if entry.get("error") or entry.get("failed") or entry.get("aborted"):
            return True
        state = str(entry.get("status_text") or entry.get("state") or "").lower()
        return "fail" in state or "error" in state or "abort" in state
    return False


def _network_failure_text(entry: dict[str, Any]) -> str:
    """Human-readable one-line description of a failed network request."""
    method = str(entry.get("method", "GET")).upper()
    url = str(entry.get("url", "")).strip()
    status = _to_int(entry.get("status"))
    if status > 0:
        return f"{method} {url} -> {status}"
    reason = str(entry.get("error") or entry.get("status_text") or "no response").strip()
    return f"{method} {url} -> failed ({reason})"


def derive_error_evidence(parsed: ParsedSidecars) -> list[ErrorEvidenceItem]:
    """Build timestamped error evidence from console errors and failed requests.

    Deduplicated by text and capped at ``_MAX_EVIDENCE_ITEMS`` so an error loop
    cannot bloat the bundle or explode the grounding query set.
    """
    evidence: list[ErrorEvidenceItem] = []
    seen: set[str] = set()

    def _add(t: float, source: str, text: str) -> None:
        if text in seen:
            return
        seen.add(text)
        evidence.append(ErrorEvidenceItem(t=t, source=source, text=text))  # type: ignore[arg-type]

    for entry in parsed.console_errors:
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        stack = entry.get("stack")
        if isinstance(stack, str) and stack.strip():
            text = f"{text}\n{stack.strip()}"
        _add(_to_float(entry.get("t")), "console", text)

    for entry in parsed.network:
        if not _is_failed_request(entry):
            continue
        _add(_to_float(entry.get("t")), "network", _network_failure_text(entry))

    evidence.sort(key=lambda item: item.t)
    if len(evidence) > _MAX_EVIDENCE_ITEMS:
        logger.info("Capping error evidence at %d of %d items", _MAX_EVIDENCE_ITEMS, len(evidence))
        evidence = evidence[:_MAX_EVIDENCE_ITEMS]
    return evidence


def derive_repro_steps(parsed: ParsedSidecars, *, start_n: int = 1) -> list[ReproStep]:
    """Build reproduction steps from captured click events."""
    steps: list[ReproStep] = []
    n = start_n
    for idx, click in enumerate(parsed.clicks):
        label = str(click.get("text") or click.get("selector") or "element").strip()
        label = " ".join(label.split())[:120] or "element"
        steps.append(
            ReproStep(
                n=n,
                t=_to_float(click.get("t")),
                action=f"Click '{label}'",
                evidence=[f"click:{idx}"],
                confidence=0.7,
            )
        )
        n += 1
    return steps


def environment_from(parsed: ParsedSidecars) -> dict[str, str]:
    """Derive environment metadata (browser/os/app) from the env snapshot."""
    env = parsed.env
    ua = str(env.get("ua", ""))
    url = str(env.get("url", ""))

    environment: dict[str, str] = {
        "os": _detect_os(ua),
        "browser": _detect_browser(ua),
        "app": _host_from_url(url) or "WebApp",
        "version": str(env.get("app_version") or "unknown"),
        "url": url,
        "source": "sidecar",
    }
    component = _component_from_url(url)
    if component:
        environment["component"] = component
    return environment


def _detect_browser(ua: str) -> str:
    if "Edg/" in ua:
        return "Edge"
    if "Firefox/" in ua:
        return "Firefox"
    if "Chrome/" in ua:
        return "Chrome"
    if "Safari/" in ua:
        return "Safari"
    return "unknown"


def _detect_os(ua: str) -> str:
    if "Windows" in ua:
        return "Windows"
    if "Mac OS X" in ua or "Macintosh" in ua:
        return "macOS"
    if "Android" in ua:
        return "Android"
    if "Linux" in ua:
        return "Linux"
    return "unknown"


def _host_from_url(url: str) -> str:
    if "://" not in url:
        return ""
    rest = url.split("://", 1)[1]
    return rest.split("/", 1)[0]


def _component_from_url(url: str) -> str:
    host = _host_from_url(url)
    if not host or "://" not in url:
        return ""
    path = url.split("://", 1)[1][len(host) :].strip("/")
    segments = [seg for seg in path.split("/") if seg and "?" not in seg]
    return "/".join(segments[:2]) if segments else ""


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
