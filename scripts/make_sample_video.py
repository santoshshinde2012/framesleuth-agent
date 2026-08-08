#!/usr/bin/env python3
"""Generate a small sample recording, so the quickstart can actually be run.

The README's first analysis says `-F "video=@bug.mp4"`, and the repository has
no bug.mp4 — so the very first thing a new user is told to do requires an asset
they have to go and make. Most people trying a tool for the first time do not
have a screen recording of a bug lying around, and asking them to produce one
before they have seen anything work is a good way to lose them.

Generated rather than committed. A checked-in video is a binary blob in a git
repo that everyone clones forever, and it would be the same handful of frames
this produces in under a second from a dependency the project already has.

    uv run python scripts/make_sample_video.py
    curl -F "video=@sample.mp4" http://127.0.0.1:8010/v1/analyze

The output is deliberately unremarkable: a few seconds of a moving block with a
frame counter. It exercises decode, keyframe selection and the pipeline's
plumbing end to end. It will NOT produce an interesting analysis — there is no
UI in it and no bug to find — and that is the point of it existing: it answers
"is this thing working at all", not "is the model any good".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def make_sample(path: Path, *, seconds: int = 4, fps: int = 8) -> Path:
    """Encode a tiny synthetic H.264 clip at ``path``."""
    try:
        import av
        import numpy as np
    except ImportError:  # pragma: no cover - depends on the install extras
        sys.exit(
            "PyAV and numpy are required.\n"
            "They ship with the project's own dependencies:\n"
            "    uv sync\n"
            "    uv run python scripts/make_sample_video.py"
        )

    width, height = 640, 360
    total = seconds * fps

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for i in range(total):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # A pale background so the moving block is unambiguous to a decoder
            # and to a human checking the file is not empty.
            frame[:, :] = (24, 28, 32)

            # A block that travels left to right — motion is what makes keyframe
            # selection do something rather than picking the same frame twice.
            x = int((width - 120) * (i / max(1, total - 1)))
            frame[140:220, x : x + 120] = (236, 91, 42)

            # A coarse progress bar along the bottom, so a still from any point
            # in the clip is identifiable.
            filled = int(width * (i / max(1, total - 1)))
            frame[height - 12 : height - 4, :filled] = (23, 82, 71)

            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)

    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "output",
        nargs="?",
        default="sample.mp4",
        type=Path,
        help="where to write the clip (default: sample.mp4)",
    )
    parser.add_argument("--seconds", type=int, default=4)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    path = make_sample(args.output, seconds=args.seconds, fps=args.fps)
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path} ({size_kb:.0f} KB, {args.seconds}s @ {args.fps}fps)")
    print()
    print("Submit it:")
    print(f'  curl -F "video=@{path}" http://127.0.0.1:8010/v1/analyze')


if __name__ == "__main__":
    main()
