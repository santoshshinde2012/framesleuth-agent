"""Tests for VLM client with retry/backoff and response parsing."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from framesleuth.clients.vlm import FrameAnalysisResponse, VLMClient


class TestVLMClient:
    """Test VLM client functionality."""

    @pytest.fixture
    def vlm_client(self) -> VLMClient:
        """Create a test VLM client."""
        return VLMClient(
            base_url="http://127.0.0.1:8080",
            model="Qwen/Qwen3-VL-8B-Instruct-GGUF",
            timeout_s=10.0,
            max_retries=3,
        )

    @pytest.mark.asyncio
    async def test_analyze_frame_success(self, vlm_client: VLMClient) -> None:
        """Test successful frame analysis."""
        # Create a dummy image file
        import base64
        import tempfile

        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_data)
            image_path = f.name

        try:
            with patch("aiohttp.ClientSession") as mock_session:
                # Mock successful response
                mock_resp = AsyncMock()
                mock_resp.status = 200
                mock_resp.text = AsyncMock(
                    return_value=json.dumps(
                        {
                            "choices": [
                                {
                                    "message": {
                                        "content": json.dumps(
                                            {
                                                "caption": "Error dialog visible",
                                                "ocr_text": "TypeError: cannot read property",
                                                "ui_action": None,
                                                "is_error_state": True,
                                                "reason": "Exception dialog",
                                            }
                                        )
                                    }
                                }
                            ]
                        }
                    )
                )
                session = mock_session.return_value
                session.post.return_value.__aenter__.return_value = mock_resp

                response = await vlm_client.analyze_frame(image_path, 5.0)
                assert isinstance(response, FrameAnalysisResponse)
                assert response.is_error_state is True
                assert "TypeError" in response.ocr_text
        finally:
            import os

            os.unlink(image_path)

    @pytest.mark.asyncio
    async def test_analyze_frame_retry_on_503(self, vlm_client: VLMClient) -> None:
        """Test retry logic on 503 (service overloaded)."""
        import base64
        import tempfile

        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_data)
            image_path = f.name

        try:
            with patch("aiohttp.ClientSession") as mock_session:
                # First call: 503, second call: 200
                mock_resp_503 = AsyncMock()
                mock_resp_503.status = 503

                mock_resp_200 = AsyncMock()
                mock_resp_200.status = 200
                mock_resp_200.text = AsyncMock(
                    return_value=json.dumps(
                        {
                            "choices": [
                                {
                                    "message": {
                                        "content": json.dumps(
                                            {
                                                "caption": "OK",
                                                "ocr_text": "",
                                                "ui_action": None,
                                                "is_error_state": False,
                                                "reason": None,
                                            }
                                        )
                                    }
                                }
                            ]
                        }
                    )
                )

                mock_req_503 = AsyncMock()
                mock_req_503.__aenter__.return_value = mock_resp_503
                mock_req_503.__aexit__.return_value = False

                mock_req_200 = AsyncMock()
                mock_req_200.__aenter__.return_value = mock_resp_200
                mock_req_200.__aexit__.return_value = False

                session = mock_session.return_value
                session.post.side_effect = [mock_req_503, mock_req_200]

                # Should retry and succeed
                response = await vlm_client.analyze_frame(image_path, 5.0)
                assert response.is_error_state is False
        finally:
            import os

            os.unlink(image_path)

    @pytest.mark.asyncio
    async def test_analyze_frame_does_not_retry_on_4xx(self, vlm_client: VLMClient) -> None:
        """A 4xx client error is non-retriable and surfaces immediately (no second call)."""
        import base64
        import os
        import tempfile

        from framesleuth.errors import ModelUnavailableError

        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_data)
            image_path = f.name

        try:
            with patch("aiohttp.ClientSession") as mock_session:
                mock_resp = AsyncMock()
                mock_resp.status = 400
                mock_resp.text = AsyncMock(return_value="bad request: unknown field")

                mock_req = AsyncMock()
                mock_req.__aenter__.return_value = mock_resp
                mock_req.__aexit__.return_value = False

                session = mock_session.return_value
                session.post.side_effect = [mock_req]  # only ONE response available

                with pytest.raises(ModelUnavailableError):
                    await vlm_client.analyze_frame(image_path, 5.0)
                # No retry: the client must not have asked for a second response.
                assert session.post.call_count == 1
        finally:
            os.unlink(image_path)

    @pytest.mark.asyncio
    async def test_missing_model_is_not_reported_as_unreachable(
        self, vlm_client: VLMClient
    ) -> None:
        """Ollama's 404 "model not found" must name the model, not the server.

        Surfacing it as "model server not reachable at <url>" sent operators to
        restart a server that was answering fine.
        """
        import base64
        import os
        import tempfile

        from framesleuth.errors import ModelNotLoadedError

        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_data)
            image_path = f.name

        try:
            with patch("aiohttp.ClientSession") as mock_session:
                mock_resp = AsyncMock()
                mock_resp.status = 404
                mock_resp.text = AsyncMock(
                    return_value='{"error":"model \\"qwen2.5vl\\" not found, try pulling it first"}'
                )

                mock_req = AsyncMock()
                mock_req.__aenter__.return_value = mock_resp
                mock_req.__aexit__.return_value = False

                session = mock_session.return_value
                session.post.side_effect = [mock_req]

                with pytest.raises(ModelNotLoadedError) as excinfo:
                    await vlm_client.analyze_frame(image_path, 5.0)
                assert "Qwen/Qwen3-VL-8B-Instruct-GGUF" in excinfo.value.message
                assert "not reachable" not in excinfo.value.message
        finally:
            os.unlink(image_path)

    @pytest.mark.asyncio
    async def test_a_plain_404_is_not_blamed_on_the_model(self, vlm_client: VLMClient) -> None:
        """A 404 with no model-not-found body means the URL, not the model.

        The first version of this fix triggered on `status == 404` alone. But a
        wrong base_url, a reverse proxy, or an engine that does not serve
        /v1/chat/completions at that path all answer 404 — and telling the
        operator to `ollama pull` a model that is already there is the same
        misattribution the fix was written to remove, just pointing the other
        way. ModelNotLoadedError must require positive evidence.
        """
        import base64
        import os
        import tempfile

        from framesleuth.errors import ModelNotLoadedError, ModelUnavailableError

        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_data)
            image_path = f.name

        try:
            with patch("aiohttp.ClientSession") as mock_session:
                mock_resp = AsyncMock()
                mock_resp.status = 404
                # What nginx returns when the path is wrong.
                mock_resp.text = AsyncMock(
                    return_value="<html><head><title>404 Not Found</title></head></html>"
                )

                mock_req = AsyncMock()
                mock_req.__aenter__.return_value = mock_resp
                mock_req.__aexit__.return_value = False

                session = mock_session.return_value
                session.post.side_effect = [mock_req]

                with pytest.raises(ModelUnavailableError) as excinfo:
                    await vlm_client.analyze_frame(image_path, 5.0)
                assert not isinstance(excinfo.value, ModelNotLoadedError)
        finally:
            os.unlink(image_path)

    @pytest.mark.asyncio
    async def test_parse_response_with_markdown(self, vlm_client: VLMClient) -> None:
        """Test parsing JSON response wrapped in markdown code blocks."""
        inner = json.dumps(
            {
                "caption": "Test",
                "ocr_text": "",
                "ui_action": None,
                "is_error_state": False,
                "reason": None,
            }
        )
        response_text = json.dumps(
            {"choices": [{"message": {"content": f"```json\n{inner}\n```"}}]}
        )

        parsed = vlm_client._parse_response(response_text)
        assert parsed.caption == "Test"
        assert parsed.is_error_state is False


class TestVLMClientHealth:
    """Test VLM client health check."""

    @pytest.mark.asyncio
    async def test_warmup_success(self) -> None:
        """Test successful warmup."""
        client = VLMClient(
            base_url="http://127.0.0.1:8080",
            model="Qwen/Qwen3-VL-8B-Instruct-GGUF",
        )

        with patch.object(client, "analyze_frame", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = FrameAnalysisResponse(
                caption="Test", ocr_text="", is_error_state=False
            )
            result = await client.warmup()
            assert result is True
            mock_analyze.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
