#!/usr/bin/env python3
"""Write the agent's OpenAPI schema to a file.

This is the contract seam between the agent and every client. The website
generates its TypeScript types from this file, so a drift between the two becomes
a CI diff instead of a runtime surprise a user discovers.

    python scripts/export_openapi.py                     # -> openapi.json
    python scripts/export_openapi.py --out ../framesleuth-web/src/types/openapi.json
    python scripts/export_openapi.py --check             # fail if out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Importing the app must not require a configured environment or touch the real
# bundle directory, so point storage at a scratch path before anything loads.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def build_schema() -> dict[str, object]:
    """Return the OpenAPI document for the current app."""
    from framesleuth.config import Settings
    from framesleuth.service.api import create_app

    settings = Settings(
        BUNDLE_DIR=Path("./.openapi-tmp"),
        DATABASE_PATH=Path("./.openapi-tmp/jobs.db"),
    )
    app = create_app(settings)
    schema: dict[str, object] = app.openapi()
    return schema


def main() -> int:
    """Write (or verify) the OpenAPI document."""
    parser = argparse.ArgumentParser(description="Export the Framesleuth OpenAPI schema.")
    parser.add_argument("--out", default="openapi.json", help="Destination path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the file on disk differs from the current app.",
    )
    args = parser.parse_args()

    rendered = json.dumps(build_schema(), indent=2, sort_keys=True) + "\n"
    out_path = Path(args.out)

    if args.check:
        if not out_path.exists():
            print(f"{out_path} is missing. Run: python scripts/export_openapi.py --out {out_path}")
            return 1
        if out_path.read_text(encoding="utf-8") != rendered:
            print(
                f"{out_path} is out of date with the API.\n"
                f"Regenerate it: python scripts/export_openapi.py --out {out_path}"
            )
            return 1
        print(f"{out_path} is up to date.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    paths = len(build_schema().get("paths", {}))  # type: ignore[union-attr]
    print(f"Wrote {out_path} ({paths} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
