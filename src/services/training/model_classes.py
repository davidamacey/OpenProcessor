"""A promoted (or core-pipeline) Triton model's own class names, and how
they map onto a project's class registry (projects_plan.md §5.5, owner
D1: cross-project model sharing and class numbering).

Class identity invariant: a model's dense output index never crosses a
project boundary. Only the model's own class *names* do, matched by name
against the consuming project's registry. Nothing here ever reuses a raw
model id as a registry id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.db.repos import models as repo
from src.services.training import model_registry


if TYPE_CHECKING:
    from src.clients.curation_opensearch import ClassRegistryFile


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


async def owned_model_names(slug: str) -> list[str]:
    """Every Triton model ``slug`` owns (its live promotions), private and shared alike. The one
    ownership enumeration: project delete (unload) and ``/stats`` (``promoted_models``) both
    read it."""
    return await model_registry.owned_names(slug)


async def is_model_shared(triton_name: str, consumer: str | None = None) -> bool:
    """Whether the owner shared the model, with ``consumer`` when given, with anyone otherwise.
    ``False`` for anything no project promoted, e.g. every core-pipeline / base model."""
    consumers = await model_registry.consumers(triton_name)
    return bool(consumers) if consumer is None else consumer in consumers


async def model_owner_project(triton_name: str) -> str | None:
    """The owning project's slug -- ``None`` for a model no project promoted (a core-pipeline /
    base model belongs to no project)."""
    return await model_registry.owner_slug(triton_name)


async def model_sharing_revision(triton_name: str) -> int | None:
    """The revision the sharing PUT is conditioned on (Cropwright BA-P2-1) -- ``None`` for a
    model no project promoted."""
    if await model_registry.live_model(triton_name) is None:
        return None
    return repo.sharing_revision(len(await model_registry.consumers(triton_name)))


async def model_classes(triton_name: str) -> list[ModelClass]:
    """This model's own classes, in model-output order.

    Resolution order:
    1. ``models.labels`` of the live promotion (a model promoted through this pipeline: model
       order, never the registry's).
    2. ``labels.txt`` (every base/core-pipeline model, which ships its own ``labels.txt`` and
       was never promoted through here at all).

    There is deliberately no third fallback to a stock vocabulary
    (COCO or otherwise): a model's names come from the model's own
    labels, never borrowed from another model (class identity invariant;
    the COCO special case in ``src.utils.class_names`` was removed for
    exactly this reason).
    """
    from src.utils.class_names import get_class_names

    live = await model_registry.live_model(triton_name)
    if live is not None:
        return [ModelClass(model_id=i, name=n) for i, n in enumerate(live.model.labels)]
    names = get_class_names(triton_name)
    return [ModelClass(model_id=i, name=names[i]) for i in sorted(names)]


def _match_name(model_name: str, registry_by_name: dict[str, tuple[int, str]]) -> MappingEntry:
    """Exact, then case-insensitive, name match -- never a synonym or a
    merge (those are a human decision, not something a model mapping
    applies automatically)."""
    exact = registry_by_name.get(model_name)
    if exact is not None:
        return MappingEntry(0, model_name, exact[0], exact[1], 'exact')
    lowered = model_name.lower()
    for name, (class_id, class_name) in registry_by_name.items():
        if name.lower() == lowered:
            return MappingEntry(0, model_name, class_id, class_name, 'case_insensitive')
    return MappingEntry(0, model_name, None, None, 'none')


async def model_class_mapping(
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
    for mc in await model_classes(triton_name):
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
        else await model_owner_project(triton_name),
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
    'model_sharing_revision',
    'owned_model_names',
]
