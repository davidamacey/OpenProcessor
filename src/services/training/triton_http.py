"""Triton HTTP control calls used by the YOLO26 promoter: warm-up
inference, model load and unload.

Split out of ``triton_promote.py``. :class:`TritonHttpControl` is mixed
into :class:`~src.services.training.triton_promote.TritonPromoter`, which
supplies the three attributes declared below.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx

from src.core.logging import get_logger
from src.services.training.promote_errors import TritonLoadError
from src.services.training.triton_repo import set_explicitly_unloaded


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


logger = get_logger(__name__)


class TritonHttpControl:
    """Warm-up, load and unload requests against Triton's HTTP API."""

    triton_models_dir: Path
    triton_http_url: str
    http_timeout: float

    async def _warm_up(
        self,
        triton_name: str,
        input_size: int,
        max_batch_size: int,
        on_phase: Callable[[str], None] | None = None,
    ) -> bool:
        """Throwaway inferences (blank images) so the TensorRT engine exists
        before the first real request. The accelerator builds for the shapes it
        has seen and rebuilds (~90 s) when a request arrives outside that range,
        so batch 1 alone leaves the first dynamically-batched request cold:
        warm batch 1 AND ``max_batch_size``. Best-effort: ``False`` unless every
        warm-up succeeded; the promote itself is already complete."""
        url = f'{self.triton_http_url}/v2/models/{triton_name}/infer'
        for batch in sorted({1, max_batch_size}):
            if on_phase:
                on_phase('building' if batch == 1 else 'warming')
            body = {
                'inputs': [
                    {
                        'name': 'images',
                        'shape': [batch, 3, input_size, input_size],
                        'datatype': 'FP32',
                        # ints keep the JSON body ~2.5x smaller than 0.0
                        'data': [0] * (batch * 3 * input_size * input_size),
                    }
                ]
            }
            try:
                async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                    resp = await client.post(url, json=body)
            except httpx.HTTPError as exc:
                logger.warning(
                    'train_promote_warm_up_unreachable',
                    triton_name=triton_name,
                    batch=batch,
                    error=str(exc),
                )
                return False
            if resp.status_code != 200:
                logger.warning(
                    'train_promote_warm_up_failed',
                    triton_name=triton_name,
                    batch=batch,
                    status=resp.status_code,
                    body=resp.text[:300],
                )
                return False
        return True

    async def _trigger_load(self, triton_name: str) -> bool:
        """POST ``/v2/repository/models/<name>/load`` to Triton.

        Returns True on 200, False on non-fatal errors (so the API
        caller can surface the on-disk paths even if Triton's polling
        isn't enabled — a follow-up manual ``model load`` will pick up
        the files we just wrote).
        """
        url = f'{self.triton_http_url}/v2/repository/models/{triton_name}/load'
        try:
            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                resp = await client.post(url)
        except httpx.HTTPError as exc:
            logger.warning(
                'train_promote_triton_load_unreachable',
                triton_name=triton_name,
                error=str(exc),
            )
            return False
        if resp.status_code == 200:
            await asyncio.to_thread(
                set_explicitly_unloaded, self.triton_models_dir / triton_name, False
            )
            return True
        logger.error(
            'train_promote_triton_load_failed',
            triton_name=triton_name,
            status=resp.status_code,
            body=resp.text[:500],
        )
        # 4xx is a load-config error worth surfacing; 5xx may be transient.
        if 400 <= resp.status_code < 500:
            raise TritonLoadError(
                f'triton refused load for {triton_name!r}: {resp.status_code} {resp.text[:200]}',
                status_code=502,
            )
        return False

    async def _trigger_unload(self, triton_name: str) -> bool:
        """POST ``/v2/repository/models/<name>/unload`` to Triton.

        Unlike :meth:`_trigger_load`, a connection failure here returns
        False (not a soft-pass) — see :class:`TritonUnloadError`'s
        docstring for why the caller must treat that as fatal rather than
        proceeding to delete files anyway.
        """
        url = f'{self.triton_http_url}/v2/repository/models/{triton_name}/unload'
        try:
            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                resp = await client.post(url)
        except httpx.HTTPError as exc:
            logger.warning(
                'train_unload_triton_unreachable',
                triton_name=triton_name,
                error=str(exc),
            )
            return False
        if resp.status_code == 200:
            return True
        logger.error(
            'train_unload_triton_failed',
            triton_name=triton_name,
            status=resp.status_code,
            body=resp.text[:500],
        )
        return False
