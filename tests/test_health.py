"""Tests for health check functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from framesleuth.clients.health import (
    check_coder_health,
    check_storage_health,
    check_vlm_health,
    get_health_status,
)


class TestHealthChecks:
    """Test health check functions."""

    @pytest.mark.asyncio
    async def test_vlm_health_ready(self) -> None:
        """Test VLM health check when ready."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            session = mock_session.return_value.__aenter__.return_value
            session.get.return_value.__aenter__.return_value = mock_resp

            health = await check_vlm_health("http://127.0.0.1:8080")
            assert health.status == "ready"
            assert health.name == "vlm"

    @pytest.mark.asyncio
    async def test_vlm_health_unavailable(self) -> None:
        """Test VLM health check when unavailable."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 500
            session = mock_session.return_value.__aenter__.return_value
            session.get.return_value.__aenter__.return_value = mock_resp

            health = await check_vlm_health("http://127.0.0.1:8080")
            assert health.status == "unavailable"

    @pytest.mark.asyncio
    async def test_vlm_health_degraded_when_configured_model_absent(self) -> None:
        """A reachable server that has not pulled the model must not report `ready`.

        The README sells `vlm: ready` as "recordings analyze with a real
        classification"; /v1/models answering 200 only proves the server is up.
        """
        with patch("aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={"data": [{"id": "llama3.2:1b"}]})
            session = mock_session.return_value.__aenter__.return_value
            session.get.return_value.__aenter__.return_value = mock_resp

            health = await check_vlm_health("http://127.0.0.1:11434", model="qwen2.5vl")
            assert health.status == "degraded"
            assert health.error == "model 'qwen2.5vl' not loaded"

    @pytest.mark.asyncio
    async def test_vlm_health_ready_when_model_listed_with_ollama_tag(self) -> None:
        """Ollama reports `qwen2.5vl:latest` for the bare name operators configure."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(
                return_value={"data": [{"id": "qwen2.5vl:latest"}, {"id": "llama3.2:1b"}]}
            )
            session = mock_session.return_value.__aenter__.return_value
            session.get.return_value.__aenter__.return_value = mock_resp

            health = await check_vlm_health("http://127.0.0.1:11434", model="qwen2.5vl")
            assert health.status == "ready"
            assert health.error is None

    @pytest.mark.asyncio
    async def test_vlm_health_ready_when_catalog_is_unreadable(self) -> None:
        """An engine whose catalog we can't parse is not evidence the model is gone."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(side_effect=ValueError("not json"))
            session = mock_session.return_value.__aenter__.return_value
            session.get.return_value.__aenter__.return_value = mock_resp

            health = await check_vlm_health("http://127.0.0.1:11434", model="qwen2.5vl")
            assert health.status == "ready"

    @pytest.mark.asyncio
    async def test_vlm_missing_model_makes_overall_degraded(self) -> None:
        """The /v1/healthz envelope must not say `healthy` when the VLM can't run."""
        with (
            patch("framesleuth.clients.health.check_coder_health") as mock_coder,
            patch("framesleuth.clients.health.check_storage_health") as mock_storage,
            patch("aiohttp.ClientSession") as mock_session,
        ):
            from framesleuth.clients.health import ServiceHealth

            mock_coder.return_value = ServiceHealth(name="coder", status="ready")
            mock_storage.return_value = ServiceHealth(name="storage", status="ready")
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={"models": [{"name": "llama3.2:1b"}]})
            session = mock_session.return_value.__aenter__.return_value
            session.get.return_value.__aenter__.return_value = mock_resp

            status = await get_health_status(
                vlm_url="http://127.0.0.1:11434",
                coder_url="http://127.0.0.1:11434",
                bundle_dir="/tmp",
                vlm_model="qwen2.5vl",
            )
            assert status.status == "degraded"
            assert status.services["vlm"].status == "degraded"

    @pytest.mark.asyncio
    async def test_coder_health_ready(self) -> None:
        """Test coder health check when ready."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            session = mock_session.return_value.__aenter__.return_value
            session.get.return_value.__aenter__.return_value = mock_resp

            health = await check_coder_health("http://127.0.0.1:11434")
            assert health.status == "ready"

    @pytest.mark.asyncio
    async def test_storage_health(self) -> None:
        """Test storage health check."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            health = await check_storage_health(tmpdir)
            assert health.status == "ready"
            assert health.name == "storage"

    @pytest.mark.asyncio
    async def test_overall_health_status(self) -> None:
        """Test overall health status aggregation."""
        with (
            patch("framesleuth.clients.health.check_vlm_health") as mock_vlm,
            patch("framesleuth.clients.health.check_coder_health") as mock_coder,
            patch("framesleuth.clients.health.check_storage_health") as mock_storage,
        ):
            from framesleuth.clients.health import ServiceHealth

            mock_vlm.return_value = ServiceHealth(name="vlm", status="ready")
            mock_coder.return_value = ServiceHealth(name="coder", status="ready")
            mock_storage.return_value = ServiceHealth(name="storage", status="ready")

            status = await get_health_status(
                vlm_url="http://127.0.0.1:8080",
                coder_url="http://127.0.0.1:11434",
                bundle_dir="/tmp",
                queue_depth=0,
            )

            assert status.status == "healthy"
            assert status.queue_depth == 0
            assert len(status.services) == 3

    @pytest.mark.asyncio
    async def test_coder_down_is_degraded_not_unhealthy(self) -> None:
        """Coder is off the analysis path, so its outage is DEGRADED, not UNHEALTHY."""
        with (
            patch("framesleuth.clients.health.check_vlm_health") as mock_vlm,
            patch("framesleuth.clients.health.check_coder_health") as mock_coder,
            patch("framesleuth.clients.health.check_storage_health") as mock_storage,
        ):
            from framesleuth.clients.health import ServiceHealth

            mock_vlm.return_value = ServiceHealth(name="vlm", status="ready")
            mock_coder.return_value = ServiceHealth(name="coder", status="unavailable")
            mock_storage.return_value = ServiceHealth(name="storage", status="ready")

            status = await get_health_status(
                vlm_url="http://x", coder_url="http://y", bundle_dir="/tmp"
            )
            assert status.status == "degraded"

    @pytest.mark.asyncio
    async def test_vlm_down_is_unhealthy(self) -> None:
        """A down VLM breaks analysis, so overall status is UNHEALTHY."""
        with (
            patch("framesleuth.clients.health.check_vlm_health") as mock_vlm,
            patch("framesleuth.clients.health.check_coder_health") as mock_coder,
            patch("framesleuth.clients.health.check_storage_health") as mock_storage,
        ):
            from framesleuth.clients.health import ServiceHealth

            mock_vlm.return_value = ServiceHealth(name="vlm", status="unavailable")
            mock_coder.return_value = ServiceHealth(name="coder", status="ready")
            mock_storage.return_value = ServiceHealth(name="storage", status="ready")

            status = await get_health_status(
                vlm_url="http://x", coder_url="http://y", bundle_dir="/tmp"
            )
            assert status.status == "unhealthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
