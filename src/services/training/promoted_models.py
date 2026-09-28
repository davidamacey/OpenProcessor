"""Promoted-model discovery and cross-project ownership (moved from
``src.routers.curation.models``, W4 any_domain_plan.md §4.3): so
``profile_validation.py`` (a service module) can check a region
profile's ``detector_model`` against promoted models without importing a
router module. ``src.routers.curation.models`` imports these back under
their old private names for its existing call sites.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.config.curation import get_curation_config
from src.core.logging import get_logger
from src.services.training.model_classes import is_model_shared, model_owner_project
from src.services.training.triton_promote import resolve_triton_models_dir


if TYPE_CHECKING:
    from pathlib import Path


logger = get_logger(__name__)


def project_owns_model(model_name: str) -> bool:
    """True if ``model_name`` (a ``triton_name``) belongs to the bound
    project's namespace (projects_plan.md §5.3/§5.5: ``triton_name =
    model_prefix + requested``).

    ``default``'s ``model_prefix`` stays the empty string (the one
    deliberate exception to "no default special case" -- every
    pre-projects / core-pipeline model, never namespaced, keeps
    resolving as default's own). A non-empty prefix owns exactly the
    names it produces. A namespaced name (contains ``'__'``) that isn't
    ours belongs to some other project -- registered or not, since only
    a project's own non-empty prefix ever produces one. An *unprefixed*
    name (no ``'__'`` at all) carries no project's namespace, so it is
    owned by ``default`` alone.
    """
    from src.config.projects import DEFAULT_SLUG
    from src.services.projects.registry import get_project_registry

    cfg = get_curation_config()
    # promote.json names the owner outright; the prefix rule alone would
    # hand `default` another project's model once that project is missing
    # from the registry snapshot (deleted, or a stale snapshot).
    recorded_owner = model_owner_project(model_name)
    if recorded_owner is not None and recorded_owner != cfg.project_slug:
        return False
    own_prefix = cfg.model_prefix
    if own_prefix:
        return model_name.startswith(own_prefix)
    other_prefixes = (
        record.resources.model_prefix
        for slug, record in get_project_registry().snapshot().items()
        if slug != DEFAULT_SLUG and record.resources.model_prefix
    )
    return not any(model_name.startswith(prefix) for prefix in other_prefixes)


def discover_promoted_models(models_dir: Path | None = None) -> list[dict[str, Any]]:
    """Models promoted through this pipeline that aren't one of the
    fixed core models.

    Every ``TritonPromoter.promote()`` call writes a ``promote.json``
    back-pointer into the model's repo directory -- its presence is
    exactly "this was promoted through `/curation/train/promote`",
    independent of Triton's own load state.

    Best-effort: any I/O error scanning the repo returns an empty list
    rather than failing the whole caller's response -- this is
    supplementary discovery.
    """
    from src.routers.curation.models import _core_models

    resolved_dir = models_dir if models_dir is not None else resolve_triton_models_dir()
    fixed_names = {name for name, *_ in _core_models()}
    out: list[dict[str, Any]] = []
    try:
        entries = sorted(resolved_dir.iterdir())
    except OSError:
        return out
    for entry in entries:
        if not entry.is_dir() or entry.name in fixed_names:
            continue
        # Project scoping (projects_plan.md §5.3, D1): the shared Triton
        # repo holds every project's promoted models side by side. Only
        # this project's own are listed here; another project's shared
        # ones come from discover_foreign_shared_models, on request only.
        if not project_owns_model(entry.name):
            continue
        promote_json = entry / 'promote.json'
        if not promote_json.is_file():
            continue
        try:
            meta = json.loads(promote_json.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            logger.warning('models_status_promote_json_unreadable', name=entry.name, error=str(exc))
            meta = {}
        out.append(
            {
                'name': entry.name,
                'job_id': meta.get('job_id'),
                'version': meta.get('version'),
                'promoted_at': meta.get('promoted_at'),
            }
        )
    return out


def is_promoted(triton_name: str) -> bool:
    """A promote.json exists for ``triton_name`` at all."""
    return (resolve_triton_models_dir() / triton_name / 'promote.json').is_file()


__all__ = [
    'discover_promoted_models',
    'is_model_shared',  # re-exported from model_classes for convenience
    'is_promoted',
    'model_owner_project',  # re-exported from model_classes for convenience
    'project_owns_model',
]
