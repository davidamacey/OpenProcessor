"""Synchronous Triton clients routed through the process's serving router.

The sync call sites (inference, analyze, OCR, visual search) run on worker threads, so they use
the uncounted ``TritonRouter.choose`` (the in-flight counts belong to the event loop's callers).
The router is the one the API (or job worker) pool already holds, so a serving-profile rebuild
reaches these callers too; with no pool (a script) or no active profile it is the single default
server, exactly the behaviour before routing. No ready server for the models refuses the call
(``NoServerError``).

Client reuse: a gRPC ``InferenceServerClient`` wraps one channel, and gRPC channels are
thread-safe and meant to be shared (https://grpc.io/docs/guides/performance/), so one client per
server URL is kept for the process (``TritonClientManager``, tritonclient grpc README
https://github.com/triton-inference-server/client). The wrappers here are cached per URL too.
"""

from __future__ import annotations

from functools import cache

from src.clients.fast_face_client import FastFaceClient
from src.clients.triton_client import TritonClient
from src.clients.triton_router import TritonRouter
from src.config import get_settings
from src.config.faces import require_faces
from src.config.settings import TritonModelConfig


YOLO_CLIP = (TritonModelConfig.YOLO_MODEL, TritonModelConfig.CLIP_IMAGE_MODEL)
FACE_MODELS = (TritonModelConfig.FACE_DETECT_MODEL, TritonModelConfig.ARCFACE_MODEL)


def _process_router() -> TritonRouter:
    from src.app.resources import AppResources

    pool = AppResources.async_triton_pool
    if pool is not None:
        return pool.router
    return TritonRouter.single(get_settings().triton_url)


def routed_url(model: str, *with_models: str) -> str:
    """The gRPC address of a ready server that holds every named model."""
    return _process_router().choose(model, *with_models).url


# Per-process wrapper cache, one per server URL (the channel itself is shared by the manager).
@cache
def _triton_client(url: str) -> TritonClient:
    return TritonClient(url)


@cache
def _face_client(url: str) -> FastFaceClient:
    return FastFaceClient(triton_url=url)


def routed_triton_client(model: str, *with_models: str) -> TritonClient:
    """The sync client for a ready server holding every model the call uses."""
    return _triton_client(routed_url(model, *with_models))


def routed_face_client() -> FastFaceClient:
    """The face client for a ready server holding SCRFD and ArcFace; refuses when faces are off."""
    require_faces()
    return _face_client(routed_url(*FACE_MODELS))
