"""Tests for coder client behavior and parsing."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from framesleuth.clients.coder import CoderClient, _content_to_text
from framesleuth.errors import ModelUnavailableError


class TestCoderClient:
    """Test coder model client."""

    @pytest.fixture
    def coder_client(self) -> CoderClient:
        """Create a test coder client."""
        return CoderClient(
            base_url="http://127.0.0.1:11434",
            model="qwen2.5-coder:7b",
            timeout_s=10.0,
            max_retries=3,
        )

    @pytest.mark.asyncio
    async def test_complete_returns_text(self, coder_client: CoderClient) -> None:
        """A plain string content round-trips through complete()."""
        with patch.object(coder_client, "_call_with_retry", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = json.dumps(
                {"choices": [{"message": {"content": "  a concise summary  "}}]}
            )
            text = await coder_client.complete("system", "user")
            assert text == "a concise summary"

    @pytest.mark.asyncio
    async def test_complete_handles_list_content(self, coder_client: CoderClient) -> None:
        """OpenAI-spec list-form content must map to text, not a TypeError."""
        with patch.object(coder_client, "_call_with_retry", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "text", "text": "part one "},
                                    {"type": "text", "text": "part two"},
                                ]
                            }
                        }
                    ]
                }
            )
            text = await coder_client.complete("system", "user")
            assert text == "part one part two"

    @pytest.mark.asyncio
    async def test_complete_raises_on_empty_choices(self, coder_client: CoderClient) -> None:
        """An empty choices array maps to ModelUnavailableError, not IndexError."""
        with patch.object(coder_client, "_call_with_retry", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = json.dumps({"choices": []})
            with pytest.raises(ModelUnavailableError):
                await coder_client.complete("system", "user")

    @pytest.mark.asyncio
    async def test_warmup_success(self, coder_client: CoderClient) -> None:
        """Test warmup delegates to retry call."""
        with patch.object(coder_client, "_call_with_retry", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = '{"choices":[{"message":{"content":"Ready"}}]}'
            ok = await coder_client.warmup()
            assert ok is True
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_error_is_retried(self, coder_client: CoderClient) -> None:
        """A refused connection (server loading a model) retries, then succeeds."""
        import aiohttp

        class _OkResp:
            status = 200

            async def text(self) -> str:
                return json.dumps({"choices": [{"message": {"content": "ok"}}]})

        calls = {"n": 0}

        def _post(url: str, json: dict) -> _OkResp:
            calls["n"] += 1
            if calls["n"] == 1:
                raise aiohttp.ClientConnectionError("connection refused")
            return _OkResp()

        session = AsyncMock()
        session.post = _post
        session.closed = False
        with (
            patch.object(coder_client, "_get_session", new_callable=AsyncMock) as mock_get,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_get.return_value = session
            text = await coder_client.complete("system", "user")
        assert text == "ok"
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_missing_model_is_not_reported_as_unreachable(
        self, coder_client: CoderClient
    ) -> None:
        """A 404 from a reachable server names the model, not the connection."""
        from framesleuth.errors import ModelNotLoadedError

        class _MissingModelResp:
            status = 404

            async def text(self) -> str:
                return '{"error":"model \'qwen2.5-coder:7b\' not found"}'

        session = AsyncMock()
        session.post = lambda url, json: _MissingModelResp()
        session.closed = False
        with patch.object(coder_client, "_get_session", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = session
            with pytest.raises(ModelNotLoadedError) as excinfo:
                await coder_client.complete("system", "user")
        assert "qwen2.5-coder:7b" in excinfo.value.message
        assert "not reachable" not in excinfo.value.message


def test_content_to_text_shapes() -> None:
    """All spec-legal content shapes coerce to text."""
    assert _content_to_text("plain") == "plain"
    assert _content_to_text(None) == ""
    assert _content_to_text([{"type": "text", "text": "a"}, "b"]) == "ab"
    assert _content_to_text([{"type": "image_url"}]) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
