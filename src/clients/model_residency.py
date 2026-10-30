"""Load Triton models on first use under the opt-in ingest load profile.

By default Triton preloads every model the API can call (``docker-compose.yml``). With
``OP_LOAD_PROFILE=ingest`` (and the overlay ``docker-compose.ingest-focus.yml``) it preloads only
what photo ingest calls (the detector and the PE image and text encoders, see `docker-compose.ingest-focus.yml`), which frees the GPU memory of the face,
MobileCLIP and OCR models for ``gpu_image_prep``. Every other endpoint still answers: the client
that is about to call a model first makes sure it is READY and asks Triton (explicit model
control) to load it when it is not. A load that fails fails the call; nothing is skipped silently.

Triton reloads a model that a load request names while it is loaded, so a READY model is never
asked for again, and a load that errors because another API worker's load won is accepted when the
model is READY afterwards.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Protocol

from src.config.env import env_str


PROFILES = ('full', 'ingest')


class _SyncClient(Protocol):
    def is_model_ready(self, model_name: str) -> bool: ...
    def load_model(self, model_name: str) -> None: ...


_ready: set[str] = set()
_sync_locks: dict[str, threading.Lock] = {}
_async_locks: dict[str, asyncio.Lock] = {}
_registry_lock = threading.Lock()


def on_demand() -> bool:
    """True under the ingest profile. An unknown value is refused, never read as the default."""
    value = env_str('OP_LOAD_PROFILE', 'full').lower()
    if value not in PROFILES:
        raise ValueError(f'OP_LOAD_PROFILE={value!r}: expected one of {", ".join(PROFILES)}')
    return value == 'ingest'


def ensure_loaded_sync(client: _SyncClient, model_name: str) -> None:
    if model_name in _ready or not on_demand():
        return
    with _registry_lock:
        lock = _sync_locks.setdefault(model_name, threading.Lock())
    with lock:
        if model_name in _ready:
            return
        if not client.is_model_ready(model_name):
            try:
                client.load_model(model_name)
            except Exception:
                if not client.is_model_ready(model_name):
                    raise
        _ready.add(model_name)


async def ensure_loaded(client: Any, model_name: str) -> None:
    """``client`` is a ``tritonclient.grpc.aio`` client."""
    if model_name in _ready or not on_demand():
        return
    with _registry_lock:
        lock = _async_locks.setdefault(model_name, asyncio.Lock())
    async with lock:
        if model_name in _ready:
            return
        if not await client.is_model_ready(model_name):
            try:
                await client.load_model(model_name)
            except Exception:
                if not await client.is_model_ready(model_name):
                    raise
        _ready.add(model_name)
