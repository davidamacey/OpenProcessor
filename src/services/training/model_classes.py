"""A promoted (or core-pipeline) Triton model's own class names, and how
they map onto a project's class registry (projects_plan.md §5.5, owner
D1: cross-project model sharing and class numbering).

Class identity invariant: a model's dense output index never crosses a
project boundary. Only the model's own class *names* do, matched by name
against the consuming project's registry. Nothing here ever reuses a raw
model id as a registry id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger


if TYPE_CHECKING:
    from src.clients.curation_opensearch import ClassRegistryFile


logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelClass:
    """One class a model can predict, in the model's own dense order."""

    model_id: int
    name: str


@dataclass(frozen=True)
class MappingEntry:
    """One model class matched (or not) against the consuming project's
    registry."""

    model_id: int
    model_name: str
    class_id: int | None
    class_name: str | None
    match: str  # 'exact' | 'case_insensitive' | 'none'


@dataclass(frozen=True)
class ModelClassMapping:
    model: str
    model_project: str | None
    project: str
    entries: tuple[MappingEntry, ...]
    unmapped: tuple[str, ...]
    not_covered: tuple[str, ...]


def _promote_json_path(triton_name: str) -> Any:
    from src.services.training.triton_promote import resolve_triton_models_dir

    return resolve_triton_models_dir() / triton_name / 'promote.json'


def _classes_from_promote_json(triton_name: str) -> list[ModelClass] | None:
    """``promote.json.classes`` (§5.5 #2: model order == ``labels.txt``
    order), when the model was promoted through this pipeline and its
    promote.json already carries the (newer) ``classes`` field."""
    path = _promote_json_path(triton_name)
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    classes_raw = raw.get('classes')
    if not isinstance(classes_raw, list) or not classes_raw:
        return None
    try:
        return [ModelClass(model_id=int(c['model_id']), name=str(c['name'])) for c in classes_raw]
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(
            'model_classes_promote_json_malformed', triton_name=triton_name, error=str(exc)
        )
        return None


def is_model_shared(triton_name: str) -> bool:
    """``promote.json.shared`` -- ``False`` (never shared) for anything
    without a promote.json, e.g. every core-pipeline / base model."""
    path = _promote_json_path(triton_name)
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return bool(raw.get('shared'))


def model_owner_project(triton_name: str) -> str | None:
    """``promote.json.project`` -- ``None`` for a model with no
    promote.json (a core-pipeline / base model belongs to no project)."""
    path = _promote_json_path(triton_name)
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    project = raw.get('project')
    return str(project) if project else None


def model_classes(triton_name: str) -> list[ModelClass]:
    """This model's own classes, in model-output order.

    Resolution order:
    1. ``promote.json.classes`` (a model promoted through this pipeline,
       post the §5.5 fix -- model order, never the registry's).
    2. ``labels.txt`` (older promotes without a ``classes`` field, and
       every base/core-pipeline model, which ships its own ``labels.txt``
       and was never promoted through here at all).

    There is deliberately no third fallback to a stock vocabulary
    (COCO or otherwise): a model's names come from the model's own
    labels, never borrowed from another model (class identity invariant;
    the COCO special case in ``src.utils.class_names`` was removed for
    exactly this reason).
    """
    from src.utils.class_names import get_class_names

    from_promote = _classes_from_promote_json(triton_name)
    if from_promote is not None:
        return from_promote
    names = get_class_names(triton_name)
    return [ModelClass(model_id=i, name=names[i]) for i in sorted(names)]


def _match_name(model_name: str, registry_by_name: dict[str, tuple[int, str]]) -> MappingEntry:
    """Exact, then case-insensitive, name match -- never a synonym or a
    merge (those are a human/W10 decision, not something a model mapping
    applies automatically)."""
    exact = registry_by_name.get(model_name)
    if exact is not None:
        return MappingEntry(0, model_name, exact[0], exact[1], 'exact')
    lowered = model_name.lower()
    for name, (class_id, class_name) in registry_by_name.items():
        if name.lower() == lowered:
            return MappingEntry(0, model_name, class_id, class_name, 'case_insensitive')
    return MappingEntry(0, model_name, None, None, 'none')


def model_class_mapping(
    triton_name: str,
    registry: ClassRegistryFile,
    *,
    project: str,
    model_project: str | None = None,
) -> ModelClassMapping:
    """This model's classes matched by name against ``registry`` (the
    *consuming* project's registry -- runs for every model, own-project
    included, per §5.5 #4: a class renamed since training shows up as
    unmapped in its own project too)."""
    registry_by_name = {
        c.class_name: (c.class_id, c.class_name) for c in registry.classes if not c.deprecated
    }
    entries: list[MappingEntry] = []
    for mc in model_classes(triton_name):
        matched = _match_name(mc.name, registry_by_name)
        entries.append(
            MappingEntry(mc.model_id, mc.name, matched.class_id, matched.class_name, matched.match)
        )
    unmapped = tuple(e.model_name for e in entries if e.match == 'none')
    mapped_class_names = {e.class_name for e in entries if e.class_name}
    not_covered = tuple(
        c.class_name
        for c in registry.classes
        if not c.deprecated and c.class_name not in mapped_class_names
    )
    return ModelClassMapping(
        model=triton_name,
        model_project=model_project
        if model_project is not None
        else model_owner_project(triton_name),
        project=project,
        entries=tuple(entries),
        unmapped=unmapped,
        not_covered=not_covered,
    )


__all__ = [
    'MappingEntry',
    'ModelClass',
    'ModelClassMapping',
    'is_model_shared',
    'model_class_mapping',
    'model_classes',
    'model_owner_project',
]
