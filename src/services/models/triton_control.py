"""
Triton Model Control Service.

Provides interface for dynamic model loading/unloading via Triton's HTTP API.
"""

import asyncio
import logging
import time
from typing import Any

import httpx

from src.config import get_settings
from src.utils.upstream_errors import UpstreamError


logger = logging.getLogger(__name__)


class ModelReloadError(RuntimeError):
    """A loaded model could not be unloaded safely, or its state could not be read."""


async def _is_ready(client: httpx.AsyncClient, base_url: str, model_name: str) -> bool:
    """KServe v2 ``GET /v2/models/<name>/ready``: 200 is ready, a 4xx is not. Anything else
    (a 5xx, a dropped connection) is unknown and refuses the operation."""
    try:
        response = await client.get(f'{base_url}/v2/models/{model_name}/ready')
    except httpx.HTTPError as e:
        raise ModelReloadError(f'cannot tell whether {model_name} is loaded: {e}') from e
    if response.status_code == 200:
        return True
    if 400 <= response.status_code < 500:
        return False
    raise ModelReloadError(
        f'cannot tell whether {model_name} is loaded: HTTP {response.status_code}'
    )


async def load_model_safely(
    client: httpx.AsyncClient,
    base_url: str,
    model_name: str,
    *,
    unload_timeout: float = 30.0,
    poll_interval: float = 0.25,
) -> httpx.Response:
    """Load ``model_name``; a READY model is unloaded first and awaited, then loaded.

    Triton's model management guide says that in explicit mode a loaded model "should be
    explicitly unloaded prior to the updated version being loaded". A reload in place after the
    model's files were rewritten crashed the server (SIGSEGV in the TensorRT backend), while
    unload, wait, load never did. For that one model the cost is a short window in which it is
    not ready; requests in the window are refused by the server (4xx), not crashed.

    Another worker may reload the same model at once: Triton serialises load and unload per
    model, an unload that finds the model gone is accepted, and each sequence ends with a load.

    Raises:
        ModelReloadError: the state is unknown, or the model is still ready after
            ``unload_timeout`` seconds (no load is sent then).
    """
    if await _is_ready(client, base_url, model_name):
        await client.post(f'{base_url}/v2/repository/models/{model_name}/unload')
        deadline = time.monotonic() + unload_timeout
        while await _is_ready(client, base_url, model_name):
            if time.monotonic() >= deadline:
                raise ModelReloadError(
                    f'{model_name} is still ready {unload_timeout:g}s after its unload; not loading'
                )
            await asyncio.sleep(poll_interval)
    return await client.post(f'{base_url}/v2/repository/models/{model_name}/load')


class TritonControlService:
    """Service for controlling Triton model repository."""

    def __init__(self) -> None:
        settings = get_settings()
        host = settings.triton_host
        self.http_url = f'http://{host}:8000'
        self.timeout = 60.0

    async def load_model(self, model_name: str) -> tuple[bool, str]:
        """Load a model into Triton."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await load_model_safely(client, self.http_url, model_name)

                if response.status_code == 200:
                    logger.info(f'Loaded model {model_name} into Triton')
                    return True, f'Model {model_name} loaded successfully'

                logger.warning(f'Failed to load {model_name}: {response.text}')
                return False, f'Load failed: {response.text}'

        except ModelReloadError as e:
            return False, f'Load refused: {e}'
        except httpx.TimeoutException:
            return False, f'Timeout loading model {model_name}'
        except httpx.ConnectError:
            return False, 'Cannot connect to Triton server'
        except httpx.HTTPError as e:
            return False, f'Load error: {e}'

    async def unload_model(self, model_name: str) -> tuple[bool, str]:
        """Unload a model from Triton."""
        endpoint = f'{self.http_url}/v2/repository/models/{model_name}/unload'

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint)

                if response.status_code == 200:
                    logger.info(f'Unloaded model {model_name} from Triton')
                    return True, f'Model {model_name} unloaded successfully'

                return False, f'Unload failed: {response.text}'

        except httpx.HTTPError as e:
            return False, f'Unload error: {e}'

    async def get_repository_index(self) -> list[dict[str, Any]]:
        """Every model in the repository with its state.

        Raises:
            UpstreamError: Triton cannot answer (unreachable, a non-2xx reply or an unreadable
                body). An outage is not an empty repository: callers that can degrade catch it.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(f'{self.http_url}/v2/repository/index')
                response.raise_for_status()
                typed: list[dict[str, Any]] = response.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.warning(f'Failed to get repository index: {e}')
            raise UpstreamError('triton', 'triton_unavailable') from e
        return typed

    async def get_model_config(self, model_name: str) -> dict[str, Any] | None:
        """Get model configuration from Triton."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f'{self.http_url}/v2/models/{model_name}/config')
                if response.status_code == 200:
                    typed: dict[str, Any] | None = response.json()
                    return typed
        except (httpx.HTTPError, ValueError) as e:
            # Model may not exist or Triton may be unavailable
            logger.debug('Could not get model config for %s: %s', model_name, e)
        return None

    async def server_ready(self) -> bool:
        """Check if Triton server is ready."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f'{self.http_url}/v2/health/ready')
                return response.status_code == 200
        except httpx.HTTPError:
            return False
