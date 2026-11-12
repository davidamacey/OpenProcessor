"""The curation loop as served steps, and the Workspace's default view.

``build_steps`` is pure: it turns :class:`WorkflowFacts` into the steps, so every step state is
tested without a database. ``gather_facts`` reads the facts: the images and class registry from
OpenSearch and the registry, the partitions, near-duplicate runs, exports, training runs and
promotions from PostgreSQL. ``workspace_default_view`` is decided only here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

from sqlalchemy import exists, func, select

from src.config.curation_accessors import images_index
from src.db.models.training import DatasetExport, Promotion, TrainingRun
from src.db.repos import near_dup as near_dup_repo
from src.db.session import project_tx
from src.schemas.curation.workflow_steps import WorkflowStep, WorkflowSteps
from src.services.clustering.assign import serving_partition_id
from src.services.clustering.partitions import list_partitions
from src.services.curation.dataset_thresholds import adequacy


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.curation.class_counts import ClassCounts


ITEM_LEVEL: Final = 'class'


@dataclass(frozen=True)
class WorkflowFacts:
    images: int = 0
    active_classes: int = 0
    hotkeys_missing: int = 0
    classes_below_floor: int = 0
    item_partitions: int = 0
    item_partition_serving: bool = False
    image_near_dup_run: bool = False
    exports: int = 0
    training_runs: int = 0
    promoted_model: bool = False


def _plural(count: int, one: str, many: str) -> str:
    return f'{count} {one if count == 1 else many}'


def build_steps(f: WorkflowFacts) -> WorkflowSteps:
    steps = [
        WorkflowStep(
            id='ingest',
            label='Add images',
            detail=f'{_plural(f.images, "image", "images")} indexed.'
            if f.images
            else 'No images yet.',
            done=f.images > 0,
            section='ingest',
        ),
        WorkflowStep(
            id='classes',
            label='Create classes from the detector',
            detail=f'{_plural(f.active_classes, "class", "classes")} in the registry.'
            if f.active_classes
            else 'The registry has no classes.',
            done=f.active_classes > 0,
            section='classes',
        ),
        WorkflowStep(
            id='cluster',
            label='Group the detections into clusters',
            detail=f'{_plural(f.item_partitions, "clustering", "clusterings")} built.'
            if f.item_partitions
            else 'No clustering has been built.',
            done=f.item_partitions > 0,
            section='workspace',
            query='view=clusters',
        ),
        WorkflowStep(
            id='activate',
            label='Activate a clustering',
            detail='A clustering is serving.'
            if f.item_partition_serving
            else 'No clustering is serving yet.',
            done=f.item_partition_serving,
            section='workspace',
            query='view=clusters',
        ),
        WorkflowStep(
            id='label',
            label='Label enough examples of every class',
            detail='Every class has enough validated examples.'
            if f.active_classes and not f.classes_below_floor
            else f'{_plural(f.classes_below_floor, "class has", "classes have")} too few '
            'validated examples.'
            if f.classes_below_floor
            else 'There are no classes to label.',
            done=f.active_classes > 0 and f.classes_below_floor == 0,
            section='workspace',
        ),
        WorkflowStep(
            id='hotkeys',
            label='Give every class a hotkey letter',
            detail=f'{_plural(f.hotkeys_missing, "class has", "classes have")} no hotkey letter.'
            if f.hotkeys_missing
            else 'Every class has a hotkey letter.',
            done=f.hotkeys_missing == 0,
            section='classes',
        ),
        WorkflowStep(
            id='near_dups',
            label='Find near-duplicate images',
            detail='Near-duplicate groups are built.'
            if f.image_near_dup_run
            else 'Near-duplicates have not been run.',
            done=f.image_near_dup_run,
            section='quality/near-duplicates',
        ),
        WorkflowStep(
            id='export',
            label='Export the dataset',
            detail=f'{_plural(f.exports, "export", "exports")} made.'
            if f.exports
            else 'Nothing has been exported.',
            done=f.exports > 0,
            section='export',
        ),
        WorkflowStep(
            id='train',
            label='Train a model',
            detail=f'{_plural(f.training_runs, "training run", "training runs")} started.'
            if f.training_runs
            else 'No model has been trained.',
            done=f.training_runs > 0,
            section='train',
        ),
    ]
    if f.promoted_model:
        steps.append(
            WorkflowStep(
                id='review_disagreements',
                label='Review where the model disagrees with the labels',
                detail='A model is promoted; its disagreements are worth a look.',
                done=False,
                section='workspace',
                query='queue=mismatches',
            )
        )
    view: Literal['gallery', 'clusters'] = 'clusters' if f.item_partition_serving else 'gallery'
    return WorkflowSteps(
        steps=steps,
        next=next((s for s in steps if not s.done), None),
        workspace_default_view=view,
    )


async def _flag(session: AsyncSession, query: Any) -> bool:
    return bool((await session.execute(select(exists(query)))).scalar_one())


async def _count(session: AsyncSession, model: Any, project: uuid.UUID) -> int:
    stmt = select(func.count()).select_from(model).where(model.project_id == project)
    return int((await session.execute(stmt)).scalar_one())


def class_facts(classes: list[Any], counts: ClassCounts) -> tuple[int, int, int]:
    """``(active classes, without a hotkey letter, below the validated floor)``."""
    active = [c for c in classes if not c.deprecated and getattr(c, 'merged_into', None) is None]
    missing = sum(1 for c in active if not getattr(c, 'hotkey_letter', None))
    below = sum(1 for c in active if adequacy(counts.validated.get(c.class_id, 0)) == 'block')
    return len(active), missing, below


async def gather_facts(
    session: AsyncSession,
    project: uuid.UUID,
    opensearch: Any,
    classes: list[Any],
    counts: ClassCounts,
) -> WorkflowFacts:
    images = int(
        (await opensearch.count(index=images_index(), body={'query': {'match_all': {}}}))['count']
    )
    active, missing, below = class_facts(classes, counts)
    async with project_tx(session, project):
        partitions = await list_partitions(session, target='item', limit=1)
        serving = await serving_partition_id(
            session, project=project, target='item', level=ITEM_LEVEL
        )
        return WorkflowFacts(
            images=images,
            active_classes=active,
            hotkeys_missing=missing,
            classes_below_floor=below,
            item_partitions=len(partitions),
            item_partition_serving=serving is not None,
            image_near_dup_run=await near_dup_repo.current_run(session, project, 'image')
            is not None,
            exports=await _count(session, DatasetExport, project),
            training_runs=await _count(session, TrainingRun, project),
            promoted_model=await _flag(
                session,
                select(Promotion.id).where(
                    Promotion.project_id == project, Promotion.superseded_at.is_(None)
                ),
            ),
        )
