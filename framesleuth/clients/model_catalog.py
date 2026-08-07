"""Shared helpers for reasoning about *which* model a server has.

The VLM/coder clients and the health check all need the same two judgements:
"does this server list the model we are configured to use?" and "is this
error the server telling us the model is missing?". Both are engine-specific
trivia (Ollama tags, two different catalog payload shapes), so they live in one
place rather than being re-guessed at each call site.
"""

import re
from typing import Any

# Ollama answers a request for an unpulled model with 404 and a body like
# `{"error":"model \"qwen2.5vl\" not found, try pulling it first"}`; llama.cpp
# and vLLM phrase it as "model not found"/"does not exist". Matching the text
# catches servers that return 400 instead of 404 for the same condition.
_MISSING_MODEL_RE = re.compile(
    r"model[^\n]{0,80}?(?:not found|does not exist|is not available|not loaded)",
    re.IGNORECASE,
)


def looks_like_missing_model(body: str) -> bool:
    """Whether an error body says the *model* is missing (not the server).

    Args:
        body: Response body text from the model server.

    Returns:
        True when the body reads as a model-not-found error.
    """
    return bool(_MISSING_MODEL_RE.search(body))


def model_names_match(configured: str, available: str) -> bool:
    """Whether a catalog entry refers to the configured model.

    Ollama reports fully-tagged names (``qwen2.5vl:latest``) while operators
    configure the bare name (``qwen2.5vl``) — the documented quick-start config
    does exactly this — so an untagged name on either side matches any tag.
    Two *different* explicit tags do not match: ``qwen2.5vl:7b`` is not
    ``qwen2.5vl:32b``.

    Args:
        configured: Model identifier from settings.
        available: Model identifier reported by the server.

    Returns:
        True when the two identifiers refer to the same model.
    """
    conf_base, _, conf_tag = configured.strip().lower().partition(":")
    avail_base, _, avail_tag = available.strip().lower().partition(":")
    if not conf_base or conf_base != avail_base:
        return False
    return not conf_tag or not avail_tag or conf_tag == avail_tag


def extract_model_names(payload: Any) -> list[str] | None:
    """Pull model identifiers out of a catalog response.

    Handles both shapes the supported engines serve: OpenAI-compatible
    ``/v1/models`` (``{"data": [{"id": ...}]}``) and Ollama's native
    ``/api/tags`` (``{"models": [{"name": ...}]}``).

    Returns:
        The list of names (possibly empty — that is real evidence the server
        has nothing loaded), or ``None`` when the payload is not a catalog we
        recognize. ``None`` means "could not enumerate", which callers must not
        confuse with "the model is absent".
    """
    if not isinstance(payload, dict):
        return None
    entries = payload.get("data")
    key = "id"
    if not isinstance(entries, list):
        entries = payload.get("models")
        key = "name"
    if not isinstance(entries, list):
        return None
    names: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            value = entry.get(key) or entry.get("model") or entry.get("name") or entry.get("id")
        else:
            value = entry
        if isinstance(value, str) and value.strip():
            names.append(value)
    return names
