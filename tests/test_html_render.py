"""Tests for the HTML -> video/gif render options and input guards.

These cover the deterministic, dependency-free surface: option normalization
(clamping + format validation) and the empty-input guard. The actual Chromium
recording / ffmpeg encode is an optional capability exercised in integration,
not here, so the unit suite stays fast and runs without Playwright or ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from framesleuth.pipeline.html_render import (
    SUPPORTED_FORMATS,
    HtmlRenderError,
    RenderOptions,
    _auto_install_enabled,
    _frame_count,
    _gif_args,
    _mp4_args,
    _webm_args,
    render_availability,
    render_html,
)


def test_auto_install_browser_defaults_on_and_is_opt_out(monkeypatch) -> None:
    """Chromium auto-downloads on first render unless explicitly disabled."""
    monkeypatch.delenv("FRAMESLEUTH_AUTO_INSTALL_BROWSER", raising=False)
    assert _auto_install_enabled() is True
    for off in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("FRAMESLEUTH_AUTO_INSTALL_BROWSER", off)
        assert _auto_install_enabled() is False
    monkeypatch.setenv("FRAMESLEUTH_AUTO_INSTALL_BROWSER", "1")
    assert _auto_install_enabled() is True


def test_defaults_are_sane() -> None:
    opts = RenderOptions.normalized()
    assert opts.fmt == "mp4"
    assert opts.duration_s == 5.0
    assert opts.fps == 30
    assert opts.width == 1280
    assert opts.height == 720


def test_normalized_clamps_out_of_range_values() -> None:
    """A caller cannot ask for a 120fps, ten-minute render; 4K is the resolution cap."""
    opts = RenderOptions.normalized(
        fmt="webm", duration_s=600.0, fps=240, width=99999, height=99999
    )
    assert opts.duration_s == 30.0  # capped at _MAX_DURATION_S
    assert opts.fps == 60  # capped at _MAX_FPS
    assert opts.width == 3840  # capped at _MAX_WIDTH (4K)
    assert opts.height == 2160  # capped at _MAX_HEIGHT (4K)


def test_normalized_allows_1080p() -> None:
    """Full-quality 1080p exports pass through unclamped."""
    opts = RenderOptions.normalized(width=1920, height=1080)
    assert (opts.width, opts.height) == (1920, 1080)


def test_normalized_clamps_below_minimums() -> None:
    opts = RenderOptions.normalized(duration_s=0.0, fps=1, width=1, height=1)
    assert opts.duration_s == 0.5
    assert opts.fps == 5
    assert opts.width == 64
    assert opts.height == 64


def test_normalized_is_case_insensitive_on_format() -> None:
    assert RenderOptions.normalized(fmt="MP4").fmt == "mp4"
    assert RenderOptions.normalized(fmt="GIF").fmt == "gif"


@pytest.mark.parametrize("bad", ["tiff", "avi", "mov", "mp5"])
def test_normalized_rejects_unsupported_format(bad: str) -> None:
    with pytest.raises(HtmlRenderError):
        RenderOptions.normalized(fmt=bad)


def test_normalized_empty_format_falls_back_to_default() -> None:
    """An empty/None format is treated as the default (mp4), not an error."""
    assert RenderOptions.normalized(fmt="").fmt == "mp4"


def test_supported_formats_are_the_three_advertised() -> None:
    assert set(SUPPORTED_FORMATS) == {"mp4", "gif", "webm"}


def test_options_are_immutable() -> None:
    opts = RenderOptions.normalized()
    with pytest.raises((AttributeError, TypeError)):
        opts.fps = 99  # type: ignore[misc]


async def test_render_html_rejects_empty_html(tmp_path: Path) -> None:
    """Empty/whitespace HTML fails fast before any Chromium/ffmpeg work."""
    opts = RenderOptions.normalized()
    for bad in ("", "   ", "\n\t"):
        with pytest.raises(HtmlRenderError):
            await render_html(bad, opts, tmp_path / "out")


def test_frame_count_is_duration_times_fps_min_one() -> None:
    """Frame-by-frame capture materializes duration x fps frames (at least one)."""
    assert _frame_count(5.0, 30) == 150
    assert _frame_count(2.0, 60) == 120
    assert _frame_count(0.0, 30) == 1  # never zero frames
    assert _frame_count(0.5, 24) == 12


def test_mp4_args_are_color_correct_and_frame_accurate() -> None:
    """The MP4 encode preserves color (yuv420p/bt709), is near-lossless, web-ready."""
    args = _mp4_args("ffmpeg", "/f/%05d.png", 30, Path("/out/render.mp4"))
    assert args[:5] == ["ffmpeg", "-y", "-framerate", "30", "-i"]
    assert "libx264" in args
    assert "yuv420p" in args  # broad-compatibility pixel format
    assert "bt709" in args  # correct color primaries/transfer/space
    assert "+faststart" in args  # immediate streaming/seeking
    # Near-lossless quality so exported colors match the source.
    assert args[args.index("-crf") + 1] == "16"


def test_webm_args_use_vp9_lossless_ish() -> None:
    args = _webm_args("ffmpeg", "/f/%05d.png", 24, Path("/out/render.webm"))
    assert "libvpx-vp9" in args
    assert args[args.index("-crf") + 1] == "24"


def test_gif_args_use_per_clip_palette() -> None:
    args = _gif_args("ffmpeg", "/f/%05d.png", 30, 1280, Path("/out/render.gif"))
    vf = args[args.index("-vf") + 1]
    assert "palettegen" in vf and "paletteuse" in vf
    assert "fps=25" in vf  # clamped to the GIF fps ceiling


def test_render_availability_reports_a_stable_shape() -> None:
    """Availability probe is side-effect free and always returns the full shape."""
    info = render_availability()
    for key in ("playwright", "chromium", "ffmpeg", "python", "ready", "hint"):
        assert key in info
    assert isinstance(info["playwright"], bool)
    assert isinstance(info["chromium"], bool)
    assert isinstance(info["ffmpeg"], bool)
    assert isinstance(info["ready"], bool)
    assert isinstance(info["python"], str) and info["python"]
    # `ready` is only true when every dependency is present; otherwise a hint exists.
    assert info["ready"] == (info["playwright"] and info["chromium"] and info["ffmpeg"])
    if not info["ready"]:
        assert isinstance(info["hint"], str) and info["hint"]


def test_hint_does_not_ask_for_a_step_the_server_performs_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Don't tell operators to run `playwright install chromium` when we auto-run it.

    The README states plainly that "the Chromium build downloads automatically on
    your first render, so there's no separate `playwright install chromium`
    step", and then points operators at this hint to diagnose a failed render.
    The hint said the opposite. Two sources of truth disagreeing about a setup
    step is how someone concludes the docs are stale and stops trusting them.

    Both directions are pinned: silent when the server handles it, explicit when
    auto-install is switched off and it genuinely is the operator's job.
    """
    monkeypatch.setattr("framesleuth.pipeline.html_render._chromium_installed", lambda: False)
    monkeypatch.setattr(
        "framesleuth.pipeline.html_render.shutil.which", lambda _: "/usr/bin/ffmpeg"
    )
    monkeypatch.setattr("framesleuth.pipeline.html_render.metadata.version", lambda _: "1.49.0")

    monkeypatch.delenv("FRAMESLEUTH_AUTO_INSTALL_BROWSER", raising=False)
    auto = render_availability()
    assert auto["ready"] is False
    # The distinction is the FRAMING, not the mere mention: "Run: <cmd>" is an
    # instruction, and naming the command inside the opt-out parenthetical is
    # reference. Only the first one contradicts the README.
    assert (
        "Run: playwright install chromium" not in auto["hint"]
    ), "instructed a manual step the server was about to perform on its own"
    assert "no action needed" in auto["hint"], "an unready render must still say why"

    monkeypatch.setenv("FRAMESLEUTH_AUTO_INSTALL_BROWSER", "0")
    manual = render_availability()
    assert (
        "Run: playwright install chromium" in manual["hint"]
    ), "with auto-install off, the operator must be told to run it"
