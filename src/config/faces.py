"""The deployment-wide face switch (#379).

Read live on every call (no module constant), like ``OP_SEMANTIC_SEARCH_ENABLED``,
so a test can ``monkeypatch.setenv`` it. A malformed value raises ``EnvError``
rather than guessing: an unreadable privacy switch must not default to on.
"""

from __future__ import annotations

from src.config.env import env_bool
from src.config.settings import TritonModelConfig


FACES_ENV = 'OP_FACES_ENABLED'


class FacesDisabledError(RuntimeError):
    """A face operation was asked for while OP_FACES_ENABLED=false."""

    def __init__(self) -> None:
        super().__init__(f'face models are off in this deployment ({FACES_ENV}=false)')


def faces_enabled() -> bool:
    return env_bool(FACES_ENV, default=True)


def require_faces() -> None:
    """Raise ``FacesDisabledError`` when faces are off. A plain zero-argument callable so
    a router can use it as a FastAPI dependency while this module stays free of fastapi."""
    if not faces_enabled():
        raise FacesDisabledError


def face_model_names() -> frozenset[str]:
    return frozenset({TritonModelConfig.FACE_DETECT_MODEL, TritonModelConfig.ARCFACE_MODEL})
