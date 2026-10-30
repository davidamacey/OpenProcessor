"""Which Triton models the deployment requires, and whether they are READY.

``/health`` and ``/ready`` use this so an install whose startup model failed to
load (an empty ``models/<name>/1/``, a missing engine) reports not ready and
names the model, instead of looking healthy because the Triton process is live.
"""

from __future__ import annotations

from typing import Any

from src.config import get_settings
from src.config.env import env_str


def required_model_names() -> list[str]:
    """Models whose absence makes the API unable to serve its core routes.

    ``OP_REQUIRED_MODELS`` (comma list) replaces the default. The default is
    the always-exported inference set plus the PE image encoder that curation
    search and scoring need; OCR is left out because the minimal profile skips it.
    """
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
    ]


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
