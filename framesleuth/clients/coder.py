"""OpenAI-compatible coder model client with engine-agnostic abstraction.

Supports Ollama, llama.cpp, and vLLM via OpenAI API.
Per ADR-001: engine selection is configuration, not code.
"""

import asyncio
import inspect
import json
from typing import Any

import aiohttp

from framesleuth.clients.model_catalog import looks_like_missing_model
from framesleuth.errors import ModelNotLoadedError, ModelUnavailableError
from framesleuth.logging_config import get_logger

logger = get_logger("clients.coder")


async def _resolve_response(request: Any) -> Any:
    """Resolve aiohttp request object or coroutine into a response object."""
    if inspect.isawaitable(request):
        return await request
    return request


def _content_to_text(content: Any) -> str:
    """Coerce an OpenAI-style ``message.content`` into plain text.

    The spec allows ``content`` to be a string *or* a list of typed parts;
    a proxy or newer server may return the list form, and ``null`` appears on
    tool-call-only replies. All of them must map to text, not a TypeError.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return "" if content is None else str(content)


class CoderClient:
    """OpenAI-compatible coder client (Qwen2.5-Coder).

    Handles:
    - Engine-agnostic OpenAI-compatible HTTP
    - Retry with backoff
    - Structured output
    - Context window management
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        """Initialize coder client.

        Args:
            base_url: OpenAI-compatible API base URL.
            model: Model identifier.
            timeout_s: Request timeout in seconds.
            max_retries: Maximum number of retries.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a shared, reused HTTP session (created lazily)."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def aclose(self) -> None:
        """Close the shared HTTP session, if open."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> "CoderClient":
        """Enter an async context that closes the session on exit."""
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Close the shared session when leaving the context."""
        await self.aclose()

    async def complete(
        self,
        system_prompt: str,
        user_content: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.4,
    ) -> str:
        """Generic text completion: run a system+user chat turn, return the text.

        Used for synthesis tasks (e.g. recording summaries) where the caller
        supplies the full system prompt. Raises ``ModelUnavailableError`` if the
        model is unreachable or the response can't be parsed, so callers can
        degrade gracefully.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        raw = await self._call_with_retry(payload)
        try:
            data = json.loads(raw)
            return _content_to_text(data["choices"][0]["message"]["content"]).strip()
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            logger.error(f"Failed to parse completion response: {exc}")
            raise ModelUnavailableError("Coder", self.base_url) from exc

    async def _call_with_retry(self, payload: dict[str, Any]) -> str:
        """Call coder with exponential backoff retry.

        Args:
            payload: Request payload.

        Returns:
            Raw response text.

        Raises:
            ModelUnavailableError: If retries exhausted.
        """
        url = f"{self.base_url}/v1/chat/completions"

        for attempt in range(self.max_retries):
            try:
                session = await self._get_session()
                resolved = await _resolve_response(session.post(url, json=payload))
                if hasattr(resolved, "__aenter__"):
                    async with resolved as resp:
                        text = await self._handle_status(resp, attempt)
                else:
                    text = await self._handle_status(resolved, attempt)
                if text is not None:
                    return text
                await asyncio.sleep(2**attempt)
            except TimeoutError:
                logger.warning(f"Coder timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise ModelUnavailableError("Coder", self.base_url) from None
            except ModelUnavailableError:
                raise
            except (aiohttp.ClientError, ConnectionError, OSError) as e:
                # A local model server that is (re)loading commonly refuses the
                # first connection — retry with backoff like a transient 5xx.
                logger.warning(
                    f"Coder connection error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise ModelUnavailableError("Coder", self.base_url) from e
            except Exception as e:
                logger.error(f"Coder request failed: {e}")
                raise ModelUnavailableError("Coder", self.base_url) from e

        raise ModelUnavailableError("Coder", self.base_url)

    async def _handle_status(self, resp: Any, attempt: int) -> str | None:
        """Map an HTTP response to text, a retry signal, or a fatal error.

        Args:
            resp: Response object exposing ``status`` and ``text()``.
            attempt: Zero-based attempt index for retry accounting.

        Returns:
            Response text on success, or ``None`` to signal a retriable 503.

        Raises:
            ModelNotLoadedError: When the server answers but the model is absent.
            ModelUnavailableError: On other non-retriable or exhausted failures.
        """
        if resp.status == 200:
            text: str = await resp.text()
            logger.info(f"Coder call successful (attempt {attempt + 1})")
            return text
        # Same retry policy as the VLM client: transient server-side failures
        # (overload, gateway, rate limit — Ollama returns 500 under load) are
        # worth retrying; 4xx client errors would fail identically every time.
        retriable = resp.status in (429, 500, 502, 503, 504)
        if retriable and attempt < self.max_retries - 1:
            logger.warning(
                f"Coder returned {resp.status} (attempt {attempt + 1}/{self.max_retries})"
            )
            return None
        if not retriable:
            body = (await resp.text())[:500]
            logger.error(f"Coder returned non-retriable {resp.status}: {body}")
            # Same distinction as the VLM client: a reachable server missing the
            # model needs `ollama pull`, not a restart.
            # NOTE: the body must actually say so. A bare 404 is ambiguous —
            # a wrong base_url, a reverse proxy, or an engine not serving
            # /v1/chat/completions at that path all return one, and reporting
            # those as "model not loaded" would repeat the exact misattribution
            # this branch exists to remove, just pointing the other way.
            if looks_like_missing_model(body):
                raise ModelNotLoadedError("Coder", self.model, self.base_url)
        raise ModelUnavailableError("Coder", self.base_url)

    async def warmup(self) -> bool:
        """Warm up coder by calling with a small prompt.

        Returns:
            True if warmup successful.
        """
        try:
            logger.info("Warming up coder...")
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Say 'Ready'."}],
                "temperature": 0.5,
                "max_tokens": 10,
            }
            await self._call_with_retry(payload)
            logger.info("Coder warmup complete")
            return True
        except Exception as e:
            logger.warning(f"Coder warmup failed: {e}")
            return False
