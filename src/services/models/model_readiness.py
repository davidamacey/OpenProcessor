"""Which Triton models the deployment requires, and whether they are READY.

``/health`` and ``/ready`` use this so an install whose startup model failed to
load (an empty ``models/<name>/1/``, a missing engine) reports not ready and
names the model, instead of looking healthy because the Triton process is live.
"""

from __future__ import annotations

from typing import Any

from src.config import get_settings
from src.config.env import env_str
from src.config.faces import face_model_names, faces_enabled


def required_model_names() -> list[str]:
    """Models whose absence makes the API unable to serve its core routes.

    ``OP_REQUIRED_MODELS`` (comma list) replaces the default. The default is
    the always-exported inference set plus the PE image encoder that curation
    search and scoring need; OCR is left out because the minimal profile skips it.
    With ``OP_FACES_ENABLED=false`` the face models are dropped from either list, so a
    list written before faces were switched off cannot hold readiness hostage.
    """
    names = _configured_or_default()
    if faces_enabled():
        return names
    face_models = face_model_names()
    return [n for n in names if n not in face_models]


def _configured_or_default() -> list[str]:
    configured = [n.strip() for n in env_str('OP_REQUIRED_MODELS').split(',') if n.strip()]
    if configured:
        return configured
    models = get_settings().models
    return [
        models.YOLO_MODEL,
        models.FACE_DETECT_MODEL,
        models.ARCFACE_MODEL,
        models.CLIP_IMAGE_MODEL,
        models.CLIP_TEXT_MODEL,
        models.PE_IMAGE_MODEL,
        models.PE_CROP_MODEL,
    ]


def crop_input_problem(config: dict[str, Any] | None, model: str, size: int) -> str | None:
    """Why the crop engine ``model`` is not built at ``size`` (``PE_CROP_SIZE``), else ``None``.

    ``config`` is the answer of ``GET /v2/models/<model>/config`` (the model configuration
    extension: the ModelConfig as JSON, int64 ``dims`` as strings, the batch axis not among
    them). A config that cannot be read, or has no ``images`` input of ``[3, H, W]``, is a
    problem: the size is not known to be right.
    """
    inputs = (config or {}).get('input') or []
    dims = next((i.get('dims') for i in inputs if i.get('name') == 'images'), None)
    if not isinstance(dims, list) or len(dims) != 3:
        return f"{model}: the server's configuration has no 'images' input of [3, H, W]"
    built = tuple(int(d) for d in dims[1:])
    if built != (size, size):
        return (
            f'{model} is built at {built[0]}x{built[1]} but PE_CROP_SIZE is {size}: '
            'set PE_CROP_SIZE to the size of the engine, or rebuild it at that size (./openprocessor models install --only pe)'
        )
    return None


def models_not_ready(
    repository_index: list[dict[str, Any]], required: list[str]
) -> list[dict[str, str]]:
    """The required models with no READY version, as ``{name, state, reason}``.

    A model Triton does not list at all is ``MISSING``.
    """
    ready: set[str] = set()
    seen: dict[str, dict[str, str]] = {}
    for entry in repository_index:
        name = str(entry.get('name', ''))
        if not name:
            continue
        if entry.get('state') == 'READY':
            ready.add(name)
        seen[name] = {
            'name': name,
            'state': str(entry.get('state') or 'UNKNOWN'),
            'reason': str(entry.get('reason') or ''),
        }
    return [
        seen.get(name, {'name': name, 'state': 'MISSING', 'reason': 'not in the Triton repository'})
        for name in required
        if name not in ready
    ]
