"""What one run of one level needs: the index, the vector field, the cuts and the guards.

Resolved once per job from the active profile and the registry (``embedding_space``), so no
reader computes a field name and a run records the exact space it grouped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.config.project_context import current_project
from src.domain.embedding_spaces import Target
from src.services.embedding_spaces.resolve import embedding_space


if TYPE_CHECKING:
    from src.services.near_duplicates.profiles import ActiveProfile

IMAGE = 'image'
ITEM = 'item'
_TARGETS = {IMAGE: Target.FRAME, ITEM: Target.ITEM}


@dataclass(frozen=True)
class LevelPlan:
    level: str
    index: str
    field: str
    space_key: str
    dedup_cosine: float
    leak_cosine: float
    k: int
    max_group_size: int
    same_detector_class: bool

    @property
    def lowest_cut(self) -> float:
        return min(self.dedup_cosine, self.leak_cosine)

    @property
    def is_item(self) -> bool:
        return self.level == ITEM

    @property
    def params(self) -> dict[str, float | int | bool]:
        """The cuts a run records (``near_dup_runs.params``)."""
        return {
            'dedup_cosine': self.dedup_cosine,
            'leak_cosine': self.leak_cosine,
            'k': self.k,
            'max_group_size': self.max_group_size,
            'same_detector_class': self.same_detector_class,
        }


async def level_plan(profile: ActiveProfile, level: str) -> LevelPlan:
    """The plan of ``level`` for the bound project. ``k`` applies to the image level only; an
    item candidate pool is already bounded by its image leak group."""
    if level not in _TARGETS:
        msg = f'level {level!r} has no project plan'
        raise ValueError(msg)
    section = profile.body.image if level == IMAGE else profile.body.item
    space = await embedding_space(_TARGETS[level], section.space_key)
    index = current_project().record.resources.indexes[space.index_role]  # type: ignore[index]
    return LevelPlan(
        level=level,
        index=index,
        field=space.vector_field,
        space_key=space.space_key,
        dedup_cosine=section.dedup_cosine,
        leak_cosine=section.leak_cosine,
        k=profile.body.image.k,
        max_group_size=section.max_group_size,
        same_detector_class=level == ITEM and profile.body.item.same_detector_class,
    )
