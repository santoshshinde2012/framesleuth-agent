"""`.env.example` must describe the settings the service actually reads.

It drifted by exactly one, and of course it was the security-relevant one:
`ALLOW_LOCAL_DEV_ORIGINS`, added when the CORS default was tightened. That
setting is the only way to reach an empty allowlist — the state the deployment
checklist tells operators to use — so an operator who could not know it existed
could not follow their own security guidance.

Documentation nobody checks is documentation that rots, so this asserts it.
"""

from __future__ import annotations

import re
from pathlib import Path

from framesleuth.config import Settings

_ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def _documented() -> set[str]:
    """Env var names assigned in the example, commented-out ones included."""
    return set(re.findall(r"^\s*#?\s*([A-Z][A-Z0-9_]+)=", _ENV_EXAMPLE.read_text("utf-8"), re.M))


def test_the_parser_found_something() -> None:
    """Sanity floor: a reformat must fail this suite, not hollow it out."""
    assert _ENV_EXAMPLE.is_file()
    assert len(_documented()) >= 40


def test_every_setting_is_documented() -> None:
    """The direction that hurts: a setting nobody can know to set."""
    missing = {n.upper() for n in Settings.model_fields} - _documented()
    assert (
        not missing
    ), f"settings the service reads but .env.example never mentions: {sorted(missing)}"


def test_the_example_invents_nothing() -> None:
    """The other direction. A documented key the service ignores is worse than
    silence, because the operator believes they configured something."""
    known = {n.upper() for n in Settings.model_fields}
    invented = _documented() - known
    assert (
        not invented
    ), f".env.example documents variables the service never reads: {sorted(invented)}"


def test_the_cors_switch_is_documented() -> None:
    """Named explicitly because it is the one that drifted, and because it is
    the only route to an allowlist the deployment checklist says you can have."""
    assert "ALLOW_LOCAL_DEV_ORIGINS" in _documented()
