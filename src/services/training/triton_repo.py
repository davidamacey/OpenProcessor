"""Triton model-repo location and the explicit-unload marker.

Split out of :mod:`src.services.training.triton_promote`.
"""

from __future__ import annotations

import os
from pathlib import Path


# Same default as src/routers/models.py — the Triton model repo mounted
# into the API container. Override at construction time for tests.
#
# This used to be a plain module-level constant, so a deployment
# that mounts the Triton repo somewhere other than /app/models (some
# deployment overlays mount it at /models) silently wrote
# promoted models into a directory Triton never sees, with no error —
# the copy + config-write both "succeed" against a path in the
# container's writable layer. Resolving OP_TRITON_MODEL_REPO here, at
# construction time rather than import time, means the env var set for
# this container is always honored, and tests can still monkeypatch
# os.environ before constructing a TritonPromoter.
DEFAULT_TRITON_MODELS_DIR = Path('/app/models')


def resolve_triton_models_dir() -> Path:
    """``OP_TRITON_MODEL_REPO``, falling back to :data:`DEFAULT_TRITON_MODELS_DIR`."""
    override = os.environ.get('OP_TRITON_MODEL_REPO')
    return Path(override) if override else DEFAULT_TRITON_MODELS_DIR


# Triton's HTTP control endpoint. The yolo-api container shares the
# triton_net network so this resolves through Docker DNS.
DEFAULT_TRITON_HTTP_URL = 'http://triton-server:8000'


def resolve_triton_http_url() -> str:
    """``OP_TRITON_HTTP_URL`` (falling back to the legacy ``TRITON_HTTP_URL``
    name if that's the only one set anywhere in this deployment), else
    :data:`DEFAULT_TRITON_HTTP_URL`.
    """
    return (
        os.environ.get('OP_TRITON_HTTP_URL')
        or os.environ.get('TRITON_HTTP_URL')
        or DEFAULT_TRITON_HTTP_URL
    )


#: Dropped in a promoted model's repo directory by ``POST /models/{name}/unload``
#: (which keeps the files): the periodic reload must not undo that unload. A
#: promote, an explicit load and an explicit reload clear it.
UNLOADED_MARKER = 'unloaded.marker'


def set_explicitly_unloaded(model_dir: Path, unloaded: bool) -> None:
    """Record (or clear) that an operator unloaded the promoted model in
    ``model_dir``. A directory without ``promote.json`` is not a promoted
    model and is left alone."""
    marker = model_dir / UNLOADED_MARKER
    if unloaded:
        if (model_dir / 'promote.json').is_file():
            marker.touch()
    else:
        marker.unlink(missing_ok=True)
