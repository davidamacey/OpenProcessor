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

from src.config.curation_accessors import images_index, items_index
from src.db.models.jobs import Job
from src.db.models.training import DatasetExport, Promotion, TrainingRun
from src.db.repos import near_dup as near_dup_repo
from src.db.session import project_tx
from src.domain import job_states as js
from src.domain.class_hotkeys import free_letters
from src.schemas.curation.workflow_steps import WorkflowStep, WorkflowSteps
from src.services.classes.registry import keymap_letters
from src.services.clustering.assign import serving_partition_id
from src.services.clustering.partitions import list_partitions
from src.services.curation import review_queries, review_tabs
from src.services.curation.audit import has_pending_sample
from src.services.curation.dataset_thresholds import HARD_MIN_CROPS_PER_CLASS, adequacy
from src.services.curation.detector_adoption import detector_census, plan_adoption
from src.services.curation.detector_vocabulary import DETECTOR_CLASS_GROUP
from src.services.curation.open_vocab_run import OPEN_VOCAB_CLASS_GROUP
from src.services.curation.region_class import REGION_CLASS_GROUP
from src.services.curation.review_tab_status import NO_AUDIT_SAMPLE


if TYPE_CHECKING:
    from collections.abc import Collection

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.curation.class_counts import ClassCounts


ITEM_LEVEL: Final = 'class'
SEEDED_CLASS_GROUPS: Final = frozenset(
    {DETECTOR_CLASS_GROUP, REGION_CLASS_GROUP, OPEN_VOCAB_CLASS_GROUP}
)


@dataclass(frozen=True)
class WorkflowFacts:
    images: int = 0
    active_classes: int = 0
    hotkeys_missing: int = 0
    hotkeys_free: int = 0
    classes_below_floor: int = 0
    item_partitions: int = 0
    item_partition_serving: bool = False
    image_near_dup_run: bool = False
    exports: int = 0
    training_runs: int = 0
    training_finished: int = 0
    training_classes_missing: int = 0
    promoted_model: bool = False
    probe_scored: bool = False
    disagreements: int = 0
    audit_pending: bool = False
    detector_classes_missing: int = 0
    detector_items_undecided: int = 0
    machine_labels_unchecked: int = 0


def _plural(count: int, one: str, many: str) -> str:
    return f'{count} {one if count == 1 else many}'


def _classes_step(f: WorkflowFacts) -> WorkflowStep:
    """Done when the registry has classes and every class the detector named is in it, with every
    item the detector named carrying one; otherwise it offers to adopt them."""
    gaps = []
    if f.detector_classes_missing:
        gaps.append(
            f'{_plural(f.detector_classes_missing, "detector class is", "detector classes are")} '
            'not in the registry'
        )
    if f.detector_items_undecided:
        gaps.append(
            f'{_plural(f.detector_items_undecided, "item has", "items have")} a detector class '
            'and no class'
        )
    if gaps:
        detail = '; '.join(gaps) + '.'
    elif f.active_classes:
        detail = f'{_plural(f.active_classes, "class", "classes")} in the registry.'
    else:
        detail = 'The registry has no classes.'
    return WorkflowStep(
        id='classes',
        label='Create classes from the detector',
        detail=detail,
        done=f.active_classes > 0 and not gaps,
        section='classes',
        action='adopt_detector_classes' if gaps else None,
    )


def _label_step(f: WorkflowFacts) -> WorkflowStep:
    """Done when every class has enough validated examples; says where the work is: the machine
    labels nobody has checked wait in the Check machine labels queue."""
    if f.active_classes and not f.classes_below_floor:
        floor = 'Every class has enough validated examples.'
    elif f.classes_below_floor:
        floor = (
            f'{_plural(f.classes_below_floor, "class has", "classes have")} too few '
            'validated examples.'
        )
    else:
        floor = 'There are no classes to label.'
    queue = review_tabs.QUEUE_LABELS['check_machine'][0]
    unchecked = f.machine_labels_unchecked
    noun = 'machine label' if unchecked == 1 else 'machine labels'
    waiting = f'{unchecked:,} {noun} to check in {queue}. ' if unchecked else ''
    return WorkflowStep(
        id='label',
        label='Label enough examples of every class',
        detail=waiting + floor,
        done=f.active_classes > 0 and f.classes_below_floor == 0,
        section='workspace',
        query='queue=check_machine' if f.machine_labels_unchecked else None,
    )


def _hotkeys_detail(f: WorkflowFacts) -> str:
    if not f.active_classes:
        return 'There are no classes to give a letter to.'
    if f.hotkeys_missing and f.hotkeys_free:
        return (
            f'{_plural(f.hotkeys_missing, "class you created has", "classes you created have")} '
            f'no hotkey letter; {_plural(f.hotkeys_free, "letter is", "letters are")} free.'
        )
    if f.hotkeys_missing:
        return 'Every free hotkey letter is in use.'
    return 'Every class you created has a hotkey letter.'


def _train_step(f: WorkflowFacts) -> WorkflowStep:
    """Done when a run has finished and its latest finished run trained every class that has
    enough trainable crops; a started, failed or partial run does not count."""
    if f.training_finished == 0:
        detail = (
            f'{_plural(f.training_runs, "training run", "training runs")} started, none finished.'
            if f.training_runs
            else 'No model has been trained.'
        )
    elif f.training_classes_missing:
        detail = (
            f'{_plural(f.training_finished, "training run", "training runs")} finished; the '
            f'latest left out {_plural(f.training_classes_missing, "class", "classes")} that '
            'now have enough trainable crops.'
        )
    else:
        detail = f'{_plural(f.training_finished, "training run", "training runs")} finished.'
    return WorkflowStep(
        id='train',
        label='Train a model',
        detail=detail,
        done=f.training_finished > 0 and f.training_classes_missing == 0,
        section='train',
    )


def _disagreements_step(f: WorkflowFacts) -> WorkflowStep:
    """After a promote: first score the items with the model (the Train page's probe), then work
    the Disagreements queue, whose count is the queue's own."""
    if not f.probe_scored:
        return WorkflowStep(
            id='probe',
            label='Find where the promoted model disagrees with your labels',
            detail='A model is promoted; run "Find disagreements with the new model" from the '
            'Train page or the Workspace More menu to fill the Disagreements queue.',
            done=False,
            section='train',
        )
    count = _plural(f.disagreements, 'item', 'items')
    return WorkflowStep(
        id='review_disagreements',
        label='Review where the model disagrees with the labels',
        detail=f'{count} in Disagreements.' if f.disagreements else 'Disagreements is empty.',
        done=f.disagreements == 0,
        section='workspace',
        query='queue=disagreements',
    )


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
        _classes_step(f),
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
        _label_step(f),
        WorkflowStep(
            id='hotkeys',
            label='Give every class a hotkey letter',
            detail=_hotkeys_detail(f),
            done=f.active_classes > 0 and (f.hotkeys_missing == 0 or f.hotkeys_free == 0),
            section='classes',
        ),
        WorkflowStep(
            id='near_dups',
            label='Find near-duplicate images',
            detail='Near-duplicate groups are built.'
            if f.image_near_dup_run
            else 'Near-duplicates have not been run.',
            done=f.image_near_dup_run,
            section='workspace',
            query='view=gallery&near_dup=collapsed',
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
        _train_step(f),
    ]
    if f.promoted_model:
        steps.append(_disagreements_step(f))
    if f.images and not f.audit_pending:
        steps.append(
            WorkflowStep(
                id='audit',
                label='Draw an audit sample',
                detail=NO_AUDIT_SAMPLE,
                done=False,
                section='quality/audit',
            )
        )
    view: Literal['gallery', 'clusters'] = 'clusters' if f.item_partition_serving else 'gallery'
    return WorkflowSteps(
        steps=steps,
        next=next((s for s in steps if not s.done), None),
        workspace_default_view=view,
    )


async def _finished_run_classes(session: AsyncSession, project: uuid.UUID) -> list[list[str]]:
    """The class names of each finished run of the project, newest first."""
    stmt = (
        select(TrainingRun.class_names)
        .join(Job, Job.id == TrainingRun.job_id)
        .where(
            TrainingRun.project_id == project,
            Job.status.in_((js.COMPLETED, js.COMPLETED_WITH_ERRORS)),
        )
        .order_by(Job.created_at.desc(), Job.id.desc())
    )
    return [list(names) for names in (await session.execute(stmt)).scalars()]


def classes_left_out(
    classes: list[Any], counts: ClassCounts, latest_finished: list[list[str]]
) -> int:
    """Active item classes with enough trainable crops that the latest finished run did not train
    (by name; 0 when no run has finished)."""
    if not latest_finished:
        return 0
    trained = set(latest_finished[0])
    return sum(
        1
        for c in classes
        if not c.deprecated
        and getattr(c, 'merged_into', None) is None
        and counts.trainable(c.class_id) >= HARD_MIN_CROPS_PER_CLASS
        and c.class_name not in trained
    )


async def _flag(session: AsyncSession, query: Any) -> bool:
    return bool((await session.execute(select(exists(query)))).scalar_one())


async def _count(session: AsyncSession, model: Any, project: uuid.UUID) -> int:
    stmt = select(func.count()).select_from(model).where(model.project_id == project)
    return int((await session.execute(stmt)).scalar_one())


def class_facts(
    classes: list[Any], counts: ClassCounts, reserved: Collection[str] = ()
) -> tuple[int, int, int, int]:
    """``(active classes, user-created ones without a hotkey letter, below the validated floor,
    free hotkey letters)``. A detector, region or open-vocabulary seed is given no letter on
    purpose, so only the other classes count as missing one."""
    active = [c for c in classes if not c.deprecated and getattr(c, 'merged_into', None) is None]
    missing = sum(
        1
        for c in active
        if not getattr(c, 'hotkey_letter', None) and c.group not in SEEDED_CLASS_GROUPS
    )
    below = sum(1 for c in active if adequacy(counts.validated.get(c.class_id, 0)) == 'block')
    taken = [c.hotkey_letter for c in classes if getattr(c, 'hotkey_letter', None)]
    return len(active), missing, below, len(free_letters(taken, reserved))


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
    active, missing, below, free = class_facts(
        classes, counts, await keymap_letters(session, project)
    )
    must, must_not, _reason = review_queries.build_tab_query(
        'check_machine', include_test=False, text=None, max_rank=None, vector_field=None
    )
    unchecked = int(
        (
            await opensearch.count(
                index=items_index(),
                body={'query': {'bool': {'must': must, 'must_not': must_not}}},
            )
        )['count']
    )
    must, must_not, _reason = review_queries.build_tab_query(
        'disagreements', include_test=False, text=None, max_rank=None, vector_field=None
    )
    disagreements = int(
        (
            await opensearch.count(
                index=items_index(),
                body={'query': {'bool': {'must': must, 'must_not': must_not}}},
            )
        )['count']
    )
    probe_scored = (
        int(
            (
                await opensearch.count(
                    index=items_index(), body={'query': {'exists': {'field': 'probe_pred_class'}}}
                )
            )['count']
        )
        > 0
    )
    plan = plan_adoption(await detector_census(opensearch), classes)
    async with project_tx(session, project):
        finished = await _finished_run_classes(session, project)
        partitions = await list_partitions(session, target='item', limit=1)
        serving = await serving_partition_id(
            session, project=project, target='item', level=ITEM_LEVEL
        )
        return WorkflowFacts(
            images=images,
            active_classes=active,
            hotkeys_missing=missing,
            hotkeys_free=free,
            classes_below_floor=below,
            item_partitions=len(partitions),
            item_partition_serving=serving is not None,
            image_near_dup_run=await near_dup_repo.current_run(session, project, 'image')
            is not None,
            exports=await _count(session, DatasetExport, project),
            training_runs=await _count(session, TrainingRun, project),
            training_finished=len(finished),
            training_classes_missing=classes_left_out(classes, counts, finished[:1]),
            probe_scored=probe_scored,
            disagreements=disagreements,
            audit_pending=await has_pending_sample(opensearch),
            detector_classes_missing=len(plan.create),
            detector_items_undecided=plan.items,
            machine_labels_unchecked=unchecked,
            promoted_model=await _flag(
                session,
                select(Promotion.id).where(
                    Promotion.project_id == project, Promotion.superseded_at.is_(None)
                ),
            ),
        )
