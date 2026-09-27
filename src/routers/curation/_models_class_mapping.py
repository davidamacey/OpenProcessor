"""Model list fields for cross-project sharing, and
``GET /models/{model_name}/class_mapping`` (projects_plan.md §5.5 #3/#4,
Cropwright delta 8).

A model's classes reach another project by NAME only:
:func:`~src.services.training.model_classes.model_class_mapping` matches
them onto the *bound* (consuming) project's registry. It runs for every
listed model, own-project included, so a class renamed since training
shows up as unmapped in its own project too.

Split out of ``models.py`` to stay under the repo's 700-LOC ratchet.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from src.config.curation import get_curation_config
from src.routers.curation._common import get_class_registry, logger, router
from src.routers.curation._config_common_models import api_error
from src.services.training.model_classes import (
    is_model_shared,
    model_class_mapping,
    model_classes,
    model_owner_project,
)
from src.services.training.triton_promote import resolve_triton_models_dir


if TYPE_CHECKING:
    from pathlib import Path

    from src.clients.curation_opensearch import ClassRegistryFile


MatchKind = Literal['exact', 'case_insensitive', 'none']

MATCH_LABELS: dict[str, str] = {
    'exact': 'Same name',
    'case_insensitive': 'Same name, different case',
    'none': 'No match',
}


class ModelClassMappingSummary(BaseModel):
    """``class_mapping`` on a model list entry: how many of the model's
    classes map onto the bound project's registry, and which do not."""

    mapped_count: int
    unmapped: list[str]


class ModelClassMappingEntry(BaseModel):
    model_id: int
    model_name: str
    class_id: int | None
    class_name: str | None
    match: MatchKind


class ModelClassMappingLabels(BaseModel):
    match: dict[str, str] = Field(default_factory=lambda: dict(MATCH_LABELS))


class ModelClassMappingResponse(BaseModel):
    """The full name mapping of one model onto the bound project."""

    model: str
    model_project: str | None
    project: str
    entries: list[ModelClassMappingEntry]
    unmapped: list[str]
    not_covered: list[str]
    labels: ModelClassMappingLabels = Field(default_factory=ModelClassMappingLabels)


def bound_registry() -> ClassRegistryFile:
    return get_class_registry().load()


def listing_fields(model_name: str, registry: ClassRegistryFile) -> dict[str, Any]:
    """``project``, ``shared`` and ``class_mapping`` for one Triton model
    list entry. ``class_mapping`` is null for a model with no class list
    (an encoder, an OCR model)."""
    summary: dict[str, Any] | None = None
    if model_classes(model_name):
        mapping = model_class_mapping(
            model_name, registry, project=get_curation_config().project_slug
        )
        summary = ModelClassMappingSummary(
            mapped_count=sum(1 for e in mapping.entries if e.match != 'none'),
            unmapped=list(mapping.unmapped),
        ).model_dump()
    return {
        'project': model_owner_project(model_name),
        'shared': is_model_shared(model_name),
        'class_mapping': summary,
    }


def _is_foreign_shared(model_name: str) -> bool:
    owner = model_owner_project(model_name)
    return (
        owner is not None
        and owner != get_curation_config().project_slug
        and is_model_shared(model_name)
    )


def discover_foreign_shared_models(models_dir: Path | None = None) -> list[dict[str, Any]]:
    """Other projects' promoted models whose owner opted in to sharing
    (``promote.json.shared``), for ``?include_other_projects=true``."""
    resolved_dir = models_dir if models_dir is not None else resolve_triton_models_dir()
    try:
        entries = sorted(resolved_dir.iterdir())
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for entry in entries:
        promote_json = entry / 'promote.json'
        if not entry.is_dir() or not promote_json.is_file() or not _is_foreign_shared(entry.name):
            continue
        try:
            meta = json.loads(promote_json.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            logger.warning('models_status_promote_json_unreadable', name=entry.name, error=str(exc))
            continue
        out.append(
            {
                'name': entry.name,
                'job_id': meta.get('job_id'),
                'promoted_at': meta.get('promoted_at'),
                'project': meta.get('project'),
            }
        )
    return out


@router.get('/models/{model_name}/class_mapping', response_model=ModelClassMappingResponse)
async def get_model_class_mapping(model_name: str) -> ModelClassMappingResponse:
    """How ``model_name``'s classes map by name onto this project's
    registry. Visible for this project's own models and base models, and
    for another project's model only once its owner shares it -- 404
    ``model_not_found`` otherwise, so an unshared model's existence and
    classes never reach another project."""
    from src.routers.curation.models import _project_owns_model

    exists = (resolve_triton_models_dir() / model_name).is_dir()
    if not exists or not (_project_owns_model(model_name) or _is_foreign_shared(model_name)):
        raise api_error(
            404,
            'model_not_found',
            f'{model_name!r} is not a model this project can use',
            project=get_curation_config().project_slug,
        )
    mapping = model_class_mapping(
        model_name, bound_registry(), project=get_curation_config().project_slug
    )
    return ModelClassMappingResponse(
        model=mapping.model,
        model_project=mapping.model_project,
        project=mapping.project,
        entries=[
            ModelClassMappingEntry(
                model_id=e.model_id,
                model_name=e.model_name,
                class_id=e.class_id,
                class_name=e.class_name,
                match=e.match,  # type: ignore[arg-type]
            )
            for e in mapping.entries
        ],
        unmapped=list(mapping.unmapped),
        not_covered=list(mapping.not_covered),
    )


__all__ = [
    'MATCH_LABELS',
    'ModelClassMappingEntry',
    'ModelClassMappingResponse',
    'ModelClassMappingSummary',
    'bound_registry',
    'discover_foreign_shared_models',
    'get_model_class_mapping',
    'listing_fields',
]
