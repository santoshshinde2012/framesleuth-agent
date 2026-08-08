"""Health checks for external dependencies.

Provides /healthz endpoint response contract and startup checks.
"""

import inspect
from typing import Any

import aiohttp
from pydantic import BaseModel

from framesleuth.clients.model_catalog import extract_model_names, model_names_match
from framesleuth.logging_config import get_logger

logger = get_logger("clients.health")


async def _resolve_response(request: Any) -> Any:
    """Resolve aiohttp request object or coroutine into a response object."""
    if inspect.isawaitable(request):
        return await request
    return request


async def _read_catalog(resp: Any) -> tuple[int, list[str] | None]:
    """Read one catalog response into its status and the model names it lists.

    Returns:
        ``(status, names)``. ``names`` is the listed model names, ``[]`` when the
        server genuinely lists none, or ``None`` when the body could not be read
        as a catalog we recognize — which is *not* evidence a model is missing.
    """
    status = int(resp.status)
    if status != 200:
        return status, None
    try:
        payload = await resp.json()
    except Exception as e:  # non-JSON body or a content-type the engine mislabels
        logger.debug(f"Model catalog body was not readable JSON: {e}")
        return status, None
    return status, extract_model_names(payload)


async def _fetch_catalog(session: Any, url: str) -> tuple[int, list[str] | None]:
    """GET a model catalog once and return ``(status, names)``.

    Transport failures propagate to the caller, which reports them as an
    unavailable service — only *parsing* problems degrade to ``names=None``.

    Args:
        session: aiohttp session (or a test double exposing ``get``).
        url: Full catalog URL.

    Returns:
        Same tuple as :func:`_read_catalog`.
    """
    resolved = await _resolve_response(session.get(url))
    if hasattr(resolved, "__aenter__"):
        async with resolved as resp:
            return await _read_catalog(resp)
    return await _read_catalog(resolved)


class ServiceHealth(BaseModel):
    """Health status of a single service."""

    name: str
    status: str  # "ready", "unavailable", "degraded"
    latency_ms: float | None = None
    error: str | None = None


class HealthStatus(BaseModel):
    """Overall health status."""

    status: str  # "healthy", "degraded", "unhealthy"
    services: dict[str, ServiceHealth]
    queue_depth: int = 0
    timestamp: str = ""


async def check_vlm_health(
    vlm_url: str, timeout_s: float = 5.0, model: str | None = None
) -> ServiceHealth:
    """Check VLM (llama.cpp) health.

    A reachable server is not the same as a usable one: with the configured
    vision model unpulled, ``/v1/models`` still answers 200 while every analysis
    fails at the first frame. The README presents ``vlm: ready`` as the signal
    that recordings analyze with a real classification, so ``ready`` is only
    reported once the model is confirmed present.

    Args:
        vlm_url: Base URL of VLM server.
        timeout_s: Timeout for health check.
        model: Configured vision model to confirm is loaded. ``None`` skips the
            check (the caller has no model to verify).

    Returns:
        Health status of VLM: ``ready``, ``degraded`` (server up, model absent),
        or ``unavailable``.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            status, names = await _fetch_catalog(session, f"{vlm_url}/v1/models")
            if status != 200:
                return ServiceHealth(name="vlm", status="unavailable", error=f"HTTP {status}")

            # No configured model to verify, or the catalog was unreadable: the
            # server answered, and claiming "not loaded" without having actually
            # enumerated anything would be the same false report in the other
            # direction.
            if not model or names is None:
                return ServiceHealth(name="vlm", status="ready")

            if not any(model_names_match(model, name) for name in names):
                # Ollama serves both catalogs and they can disagree (the native
                # one lists pulled models the OpenAI shim does not always
                # surface), so a miss is confirmed against /api/tags before the
                # operator is told anything is wrong.
                _, fallback = await _fetch_catalog(session, f"{vlm_url}/api/tags")
                if fallback is None or not any(model_names_match(model, name) for name in fallback):
                    return ServiceHealth(
                        name="vlm", status="degraded", error=f"model '{model}' not loaded"
                    )
            return ServiceHealth(name="vlm", status="ready")
    except Exception as e:
        logger.warning(f"VLM health check failed: {e}")
        return ServiceHealth(name="vlm", status="unavailable", error=str(e))


async def check_coder_health(
    coder_url: str, timeout_s: float = 5.0, model: str | None = None
) -> ServiceHealth:
    """Check coder (Ollama/llama.cpp) health.

    Same distinction the VLM check makes, for the same reason: a reachable
    server is not a usable one. This reported ``ready`` on any 200 from
    ``/api/tags`` or ``/v1/models``, so an install with the coder model unpulled
    passed its own health check and then failed at the first request — which the
    repo's own documentation calls out as the thing a health check must not do.

    Args:
        coder_url: Base URL of coder server.
        timeout_s: Timeout for health check.
        model: Configured coder model to confirm is loaded. ``None`` skips the
            check (the caller has no model to verify).

    Returns:
        Health status of coder.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Ollama's native catalog first, then the OpenAI-compatible one.
            # Both are tried because the two disagree: the native endpoint lists
            # pulled models the shim does not always surface.
            status, names = await _fetch_catalog(session, f"{coder_url}/api/tags")
            if status != 200 or names is None:
                status, names = await _fetch_catalog(session, f"{coder_url}/v1/models")
            if status != 200:
                return ServiceHealth(name="coder", status="unavailable", error=f"HTTP {status}")

            # Nothing to verify, or an unreadable catalog: the server answered,
            # and reporting "not loaded" without having enumerated anything
            # would be the same false report pointing the other way.
            if not model or names is None:
                return ServiceHealth(name="coder", status="ready")

            if not any(model_names_match(model, name) for name in names):
                _, fallback = await _fetch_catalog(session, f"{coder_url}/v1/models")
                if fallback is None or not any(model_names_match(model, name) for name in fallback):
                    return ServiceHealth(
                        name="coder", status="degraded", error=f"model '{model}' not loaded"
                    )
            return ServiceHealth(name="coder", status="ready")
    except Exception as e:
        logger.warning(f"Coder health check failed: {e}")
        return ServiceHealth(name="coder", status="unavailable", error=str(e))


async def check_storage_health(bundle_dir: str) -> ServiceHealth:
    """Check storage (filesystem) health.

    Args:
        bundle_dir: Path to bundle directory.

    Returns:
        Health status of storage.
    """
    try:
        from pathlib import Path

        path = Path(bundle_dir)
        # Try to write a temporary file
        test_file = path / ".health-check"
        test_file.write_text("ok")
        test_file.unlink()
        return ServiceHealth(name="storage", status="ready")
    except Exception as e:
        logger.warning(f"Storage health check failed: {e}")
        # Don't echo the absolute bundle path back in the response.
        from framesleuth.errors import sanitize_message

        return ServiceHealth(name="storage", status="unavailable", error=sanitize_message(str(e)))


async def get_health_status(
    vlm_url: str,
    coder_url: str,
    bundle_dir: str,
    queue_depth: int = 0,
    vlm_model: str | None = None,
    coder_model: str | None = None,
) -> HealthStatus:
    """Get overall system health status.

    Args:
        vlm_url: VLM server URL.
        coder_url: Coder server URL.
        bundle_dir: Bundle storage directory.
        queue_depth: Current job queue depth.
        vlm_model: Configured vision model, confirmed present before the VLM is
            reported ``ready``.
        coder_model: Configured coder model, confirmed the same way.

    Returns:
        Overall health status with service details.
    """
    from datetime import UTC, datetime

    vlm_health = await check_vlm_health(vlm_url, model=vlm_model)
    coder_health = await check_coder_health(coder_url, model=coder_model)
    storage_health = await check_storage_health(bundle_dir)

    services = {
        "vlm": vlm_health,
        "coder": coder_health,
        "storage": storage_health,
    }

    # Overall status: healthy if all ready. The coder endpoint is not on the
    # analysis path (summary reuses the VLM endpoint), so a coder outage is only
    # DEGRADED. A down VLM or storage is what actually breaks analysis → UNHEALTHY.
    # NB: compare service *names*, not statuses — the earlier `s != "coder"`
    # filtered status strings, which never equal "coder", so it never fired.
    critical_down = any(
        health.status == "unavailable" for name, health in services.items() if name != "coder"
    )
    if all(health.status == "ready" for health in services.values()):
        overall = "healthy"
    elif critical_down:
        overall = "unhealthy"
    else:
        overall = "degraded"

    return HealthStatus(
        status=overall,
        services=services,
        queue_depth=queue_depth,
        timestamp=datetime.now(UTC).isoformat(),
    )
