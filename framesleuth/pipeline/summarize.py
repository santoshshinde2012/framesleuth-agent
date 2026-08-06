"""Summary synthesis: turn the fused video+audio timeline into prose.

Takes the visual scenes (captions + OCR) and the audio transcript, fuses them
into one time-ordered timeline, and asks a text LLM — steered by the caller's
skill/system prompt — to produce a summary. Degrades to an empty string when
there is no evidence or the model is unavailable, so it never aborts a run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from framesleuth.clients.coder import CoderClient
from framesleuth.logging_config import get_logger
from framesleuth.pipeline.fusion import build_timeline
from framesleuth.schemas import SceneRecord, Transcript

if TYPE_CHECKING:  # pragma: no cover - typing only
    from framesleuth.pipeline.sidecars import ParsedSidecars

logger = get_logger("pipeline.summarize")

# Cap timeline events fed to the model so the prompt stays bounded on long clips.
_MAX_EVENTS = 80


def build_summary_input(
    scenes: list[SceneRecord],
    transcript: Transcript,
    user_intent: str | None,
    sidecars: ParsedSidecars | None = None,
) -> str:
    """Render the fused video+audio+sidecar timeline into the model's user message.

    ``sidecars`` is what lets the summary mention the actual console exception or
    failed request; without it the model only ever saw captions, OCR and speech.
    """
    lines: list[str] = []
    if user_intent and user_intent.strip():
        # The intent is the caller's text, already redacted upstream. It is
        # labelled as a request, not as evidence, so the model does not "cite" it.
        lines.append(f"User request: {user_intent.strip()}")
        lines.append("")
    lines.append("Recording timeline (t = seconds into the video):")
    count = 0
    for event in build_timeline(scenes, transcript, sidecars):
        text = " ".join(event.text.split()).strip()
        if not text:
            continue
        lines.append(f"- t={event.t:.1f}s [{event.kind}] {text}")
        count += 1
        if count >= _MAX_EVENTS:
            lines.append("- … (timeline truncated)")
            break
    return "\n".join(lines)


async def summarize_recording(
    scenes: list[SceneRecord],
    transcript: Transcript,
    *,
    system_prompt: str,
    user_intent: str | None,
    client: CoderClient,
    max_tokens: int = 1024,
    sidecars: ParsedSidecars | None = None,
) -> str | None:
    """Generate a summary of the recording per ``system_prompt``.

    Returns ``None`` when there was *nothing to summarize* (no scenes, no
    transcript, no sidecars) and ``""`` when the model call *failed*. The caller
    needs to tell those apart: only the second is a degraded stage — a silent
    screen recording with no UI simply has no summary to produce, and flagging
    that as degradation misreports the run's quality.
    """
    has_sidecars = bool(sidecars and (sidecars.console_errors or sidecars.network))
    if not scenes and not transcript.segments and not has_sidecars:
        return None
    content = build_summary_input(scenes, transcript, user_intent, sidecars)
    if not content.strip():
        return None
    try:
        summary = await client.complete(system_prompt, content, max_tokens=max_tokens)
    except Exception as exc:  # model/network failure must not abort the run
        logger.warning("Summary generation degraded: %s", exc)
        return ""
    logger.info("Summary generated (%d chars)", len(summary))
    return summary
