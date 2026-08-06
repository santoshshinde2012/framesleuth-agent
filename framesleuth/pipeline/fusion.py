"""Timeline fusion and citation-enforced summary assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from framesleuth.schemas import SceneRecord, Transcript

if TYPE_CHECKING:  # pragma: no cover - typing only
    from framesleuth.pipeline.sidecars import ParsedSidecars


@dataclass(frozen=True)
class TimelineEvent:
    """Unified timeline event from visual or transcript evidence."""

    t: float
    kind: str
    text: str
    citation: str


def build_timeline(
    scenes: list[SceneRecord],
    transcript: Transcript,
    sidecars: ParsedSidecars | None = None,
) -> list[TimelineEvent]:
    """Merge scene, transcript, and sidecar events in timestamp order.

    Sidecars matter here specifically because this timeline is what the
    summarizer reads. Console exceptions and failed requests used to be attached
    to the bundle as separate raw arrays and never fused, so the model writing
    the summary — the primary deliverable for a general video — literally never
    saw the strongest evidence in the recording and could not cite it.
    """
    events: list[TimelineEvent] = []

    for index, scene in enumerate(scenes):
        events.append(
            TimelineEvent(
                t=scene.t0,
                kind="scene",
                text=scene.caption,
                citation=f"frame:{index}",
            )
        )
        if scene.ocr_text.strip():
            events.append(
                TimelineEvent(
                    t=scene.t0,
                    kind="ocr",
                    text=scene.ocr_text,
                    citation=f"frame:{index}",
                )
            )

    for index, segment in enumerate(transcript.segments):
        text = segment.text.strip()
        if not text:
            continue
        events.append(
            TimelineEvent(
                t=segment.t0,
                kind="transcript",
                text=text,
                citation=f"transcript:{index}",
            )
        )

    if sidecars is not None:
        events.extend(_sidecar_events(sidecars))

    return sorted(events, key=lambda e: e.t)


def _sidecar_events(sidecars: ParsedSidecars) -> list[TimelineEvent]:
    """Turn console/network sidecar records into cited timeline events."""
    events: list[TimelineEvent] = []

    def _t(record: dict[str, Any]) -> float:
        raw = record.get("t")
        return float(raw) if isinstance(raw, int | float) else 0.0

    for index, record in enumerate(sidecars.console_errors):
        text = str(record.get("text") or record.get("message") or "").strip()
        if text:
            events.append(
                TimelineEvent(t=_t(record), kind="console", text=text, citation=f"console:{index}")
            )

    for index, record in enumerate(sidecars.network):
        url = str(record.get("url") or "").strip()
        status = record.get("status")
        method = str(record.get("method") or "").strip()
        label = " ".join(part for part in (method, url) if part) or "request"
        text = f"{label} → {status}" if status is not None else label
        events.append(
            TimelineEvent(t=_t(record), kind="network", text=text, citation=f"network:{index}")
        )

    return events


def enforce_citations(claims: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop claims without citations to avoid uncited summaries."""
    return [claim for claim in claims if claim.get("citation", "").strip()]


def summarize_timeline(events: list[TimelineEvent]) -> list[dict[str, str]]:
    """Create concise summary claims preserving evidence citations."""
    claims: list[dict[str, str]] = []
    for event in events[:20]:
        claims.append({"claim": event.text, "citation": event.citation})
    return enforce_citations(claims)
