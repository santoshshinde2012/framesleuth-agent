"""Atomic file writes for bundle artifacts.

Every artifact the API later serves is written through here. An in-place
``write_text`` leaves a *truncated* file at the canonical path if the process
dies or the disk fills mid-write — and the retention sweep, the report UI and
the MCP tools all read that path directly, so they would consume the truncated
version as if it were complete.

``gif.py`` already established this tmp-then-rename pattern for exactly that
reason ("so a partial file is never served from cache"); this is the same idea
factored out so every writer shares it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, data: str, *, encoding: str = "utf-8") -> None:
    """Write ``data`` to ``path`` so readers only ever see all of it or none.

    ``Path.replace`` is atomic within a filesystem, so a reader either sees the
    previous file or the complete new one — never a half-written mix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        with tmp_path.open("w", encoding=encoding) as handle:
            handle.write(data)
            # Flush through the OS buffer before the rename, so a crash right
            # after it cannot leave an empty file at the canonical path.
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Serialize ``payload`` and write it atomically."""
    atomic_write_text(path, json.dumps(payload, indent=indent))
