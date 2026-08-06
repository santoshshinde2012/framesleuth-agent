"""Tests for understanding and fusion pipeline modules."""

from pathlib import Path

import pytest

from framesleuth.clients.vlm import FrameAnalysisResponse
from framesleuth.pipeline.fusion import build_timeline, summarize_timeline
from framesleuth.pipeline.understand import analyze_keyframes
from framesleuth.schemas import KeyframeRef, Transcript


class FakeVLMClient:
    """Deterministic fake VLM for unit tests."""

    def __init__(self) -> None:
        self.calls = 0

    async def analyze_frame(
        self,
        image_path: str,
        timestamp: float,
        prompt_override: str | None = None,
        *,
        max_tokens: int | None = None,
        send_jpeg: bool | None = None,
    ):
        self.calls += 1
        if prompt_override:
            return FrameAnalysisResponse(
                caption="error dialog",
                ocr_text="TypeError: boom",
                ui_action="click submit",
                is_error_state=True,
                reason="exception visible",
            )
        return FrameAnalysisResponse(
            caption="error dialog",
            ocr_text="",
            ui_action="click submit",
            is_error_state=True,
            reason="exception visible",
        )


class _OneBadFrameVLM:
    """Fails on exactly one frame path; succeeds elsewhere."""

    def __init__(self, bad_file: str) -> None:
        self.bad_file = bad_file

    async def analyze_frame(
        self,
        image_path: str,
        timestamp: float,
        prompt_override: str | None = None,
        *,
        max_tokens: int | None = None,
        send_jpeg: bool | None = None,
    ) -> FrameAnalysisResponse:
        if image_path.endswith(self.bad_file):
            raise ValueError("VLM response validation failed: truncated JSON")
        return FrameAnalysisResponse(caption=f"scene at {timestamp}", ocr_text="ok text")


@pytest.mark.asyncio
async def test_one_bad_frame_is_salvaged_not_fatal(tmp_path: Path) -> None:
    """One malformed model reply drops only that frame — never the whole stage.

    Weak local models routinely truncate JSON on a single frame; the other
    frames' analyses must survive, with scenes and keyframes staying aligned.
    """
    for i in range(3):
        (tmp_path / f"{i}.png").write_bytes(b"png")
    keyframes = [KeyframeRef(index=i, t=float(i), shows="scene", file=f"{i}.png") for i in range(3)]

    scenes, analyzed = await analyze_keyframes(keyframes, tmp_path, _OneBadFrameVLM("1.png"))

    assert len(scenes) == 2
    assert [kf.file for kf in analyzed] == ["0.png", "2.png"]
    # Alignment holds: scene N came from analyzed keyframe N.
    assert scenes[0].t0 == analyzed[0].t and scenes[1].t0 == analyzed[1].t


@pytest.mark.asyncio
async def test_all_frames_failing_raises(tmp_path: Path) -> None:
    """A total VLM failure still raises so the orchestrator can degrade the stage."""
    (tmp_path / "0.png").write_bytes(b"png")
    keyframes = [KeyframeRef(index=0, t=0.0, shows="scene", file="0.png")]

    with pytest.raises(ValueError):
        await analyze_keyframes(keyframes, tmp_path, _OneBadFrameVLM("0.png"))


@pytest.mark.asyncio
async def test_understand_retries_low_ocr(tmp_path: Path) -> None:
    """Error-state low OCR should trigger one retry with override prompt."""
    frames_dir = tmp_path
    (frames_dir / "0.png").write_bytes(b"png")

    keyframes = [KeyframeRef(index=0, t=1.0, shows="error", file="0.png")]
    client = FakeVLMClient()

    scenes, analyzed = await analyze_keyframes(keyframes, frames_dir, client)

    assert len(scenes) == 1
    assert client.calls == 2
    assert "TypeError" in scenes[0].ocr_text


def test_fusion_builds_ordered_timeline_and_citations() -> None:
    """Timeline should be sorted and summary claims must preserve citation."""
    transcript = Transcript(
        segments=[Transcript.Segment(t0=0.5, t1=1.0, text="clicked button", conf=0.9)],
        words=[],
    )
    from framesleuth.schemas import SceneRecord

    scenes = [
        SceneRecord(
            t0=1.0,
            t1=1.0,
            caption="button turns red",
            ocr_text="Error 500",
            ui_action="click",
            is_error_state=True,
            reason="error visible",
        )
    ]

    timeline = build_timeline(scenes, transcript)
    assert timeline[0].t <= timeline[-1].t

    summary = summarize_timeline(timeline)
    assert summary
    assert all(item.get("citation") for item in summary)
