"""Load Triton models on first use under the opt-in ingest load profile.

By default Triton preloads every model the API can call (``docker-compose.yml``). With
``OP_LOAD_PROFILE=ingest`` (and the overlay ``docker-compose.ingest-focus.yml``) it preloads only
what photo ingest calls (the detector and the PE image and text encoders, see `docker-compose.ingest-focus.yml`), which frees the GPU memory of the face,
MobileCLIP and OCR models for ``gpu_image_prep``. Every other endpoint still answers: the client
that is about to call a model first makes sure it is READY and asks Triton (explicit model
control) to load it when it is not. A load that fails fails the call; nothing is skipped silently.

Triton reloads a model that a load request names while it is loaded, so a READY model is never
asked for again, and a load that errors because another API worker's load won is accepted when the
model is READY afterwards. Readiness is asked of Triton on every call (no process-local memo), so a
model unloaded by hand or lost to a Triton restart is loaded again; Triton serialises loads, so
concurrent callers need no lock here.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.config.env import env_str
from src.config.settings import TritonModelConfig


PROFILES = ('full', 'ingest')


class _SyncClient(Protocol):
    def is_model_ready(self, model_name: str) -> bool: ...
    def load_model(self, model_name: str) -> None: ...


# Models that call other models inside Triton (BLS): those must be loaded before the caller.
_CALLS = {
    TritonModelConfig.OCR_PIPELINE_MODEL: (
        TritonModelConfig.OCR_DET_MODEL,
        TritonModelConfig.OCR_REC_MODEL,
    ),
}


def on_demand() -> bool:
    """True under the ingest profile. An unknown value is refused, never read as the default."""
    value = env_str('OP_LOAD_PROFILE', 'full').lower()
    if value not in PROFILES:
        raise ValueError(f'OP_LOAD_PROFILE={value!r}: expected one of {", ".join(PROFILES)}')
    return value == 'ingest'


def ensure_loaded_sync(client: _SyncClient, model_name: str) -> None:
    if not on_demand():
        return
    for dep in _CALLS.get(model_name, ()):
        ensure_loaded_sync(client, dep)
    if client.is_model_ready(model_name):
        return
    try:
        client.load_model(model_name)
    except Exception:
        if not client.is_model_ready(model_name):
            raise


async def ensure_loaded(client: Any, model_name: str) -> None:
    """``client`` is a ``tritonclient.grpc.aio`` client."""
    if not on_demand():
        return
    for dep in _CALLS.get(model_name, ()):
        await ensure_loaded(client, dep)
    if await client.is_model_ready(model_name):
        return
    try:
        await client.load_model(model_name)
    except Exception:
        if not await client.is_model_ready(model_name):
            raise
