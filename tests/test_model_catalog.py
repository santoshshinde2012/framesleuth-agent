"""Tests for the shared model-catalog helpers."""

import pytest

from framesleuth.clients.model_catalog import (
    extract_model_names,
    looks_like_missing_model,
    model_names_match,
)


def test_untagged_name_matches_ollama_tag() -> None:
    """The documented config is bare (`qwen2.5vl`); Ollama reports `:latest`."""
    assert model_names_match("qwen2.5vl", "qwen2.5vl:latest")
    assert model_names_match("qwen2.5vl:latest", "qwen2.5vl")


def test_different_explicit_tags_do_not_match() -> None:
    """Matching on the base name alone would call a 32b install a 7b one."""
    assert not model_names_match("qwen2.5vl:7b", "qwen2.5vl:32b")
    assert model_names_match("qwen2.5vl:7b", "qwen2.5vl:7b")


def test_different_models_do_not_match() -> None:
    assert not model_names_match("qwen2.5vl", "llama3.2:1b")
    assert not model_names_match("", "llama3.2")


def test_extract_names_from_both_catalog_shapes() -> None:
    """OpenAI `/v1/models` and Ollama `/api/tags` disagree on keys."""
    assert extract_model_names({"data": [{"id": "a"}, {"id": "b"}]}) == ["a", "b"]
    assert extract_model_names({"models": [{"name": "a:latest"}]}) == ["a:latest"]


def test_extract_names_returns_none_for_unrecognized_payload() -> None:
    """`None` means "could not enumerate" — callers must not read it as absent."""
    assert extract_model_names(None) is None
    assert extract_model_names({"unexpected": 1}) is None
    assert extract_model_names({"data": []}) == []


@pytest.mark.parametrize(
    "body",
    [
        '{"error":"model \\"qwen2.5vl\\" not found, try pulling it first"}',
        "The model does not exist",
        "model is not available",
    ],
)
def test_missing_model_bodies_are_recognized(body: str) -> None:
    assert looks_like_missing_model(body)


def test_unrelated_error_body_is_not_a_missing_model() -> None:
    assert not looks_like_missing_model("bad request: unknown field 'response_format'")
