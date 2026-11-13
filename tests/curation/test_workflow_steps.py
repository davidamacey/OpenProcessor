"""The curation loop's next steps and the Workspace default view, decided in one place."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from src.services.curation.workflow_steps import WorkflowFacts, build_steps


EMPTY = WorkflowFacts()
DONE = WorkflowFacts(
    images=10,
    active_classes=2,
    hotkeys_missing=0,
    classes_below_floor=0,
    item_partitions=1,
    item_partition_serving=True,
    image_near_dup_run=True,
    exports=1,
    training_runs=1,
    training_finished=1,
    promoted_model=True,
    probe_scored=True,
    audit_pending=True,
)


def _step(facts: WorkflowFacts, step_id: str):
    return next(s for s in build_steps(facts).steps if s.id == step_id)


def test_a_project_with_nothing_starts_at_ingest_and_defaults_to_the_gallery() -> None:
    result = build_steps(EMPTY)
    assert result.next is not None
    assert result.next.id == 'ingest'
    assert result.next.section == 'ingest'
    assert result.workspace_default_view == 'gallery'
    assert [s.id for s in result.steps][:3] == ['ingest', 'classes', 'cluster']
    assert 'review_disagreements' not in [s.id for s in result.steps]


@pytest.mark.parametrize(
    ('step_id', 'undone', 'done'),
    [
        ('ingest', {'images': 0}, {'images': 3}),
        ('classes', {'active_classes': 0}, {'active_classes': 1}),
        ('cluster', {'item_partitions': 0}, {'item_partitions': 2}),
        (
            'activate',
            {'item_partition_serving': False},
            {'item_partitions': 1, 'item_partition_serving': True},
        ),
        ('label', {'classes_below_floor': 1}, {'active_classes': 2, 'classes_below_floor': 0}),
        (
            'hotkeys',
            {'hotkeys_missing': 2, 'hotkeys_free': 3},
            {'active_classes': 2, 'hotkeys_missing': 0},
        ),
        ('near_dups', {'image_near_dup_run': False}, {'image_near_dup_run': True}),
        ('export', {'exports': 0}, {'exports': 1}),
        ('train', {'training_finished': 0}, {'training_runs': 1, 'training_finished': 1}),
    ],
)
def test_each_step_is_done_by_its_own_fact(
    step_id: str, undone: dict[str, Any], done: dict[str, Any]
) -> None:
    assert _step(dataclasses.replace(DONE, **undone), step_id).done is False
    assert _step(dataclasses.replace(EMPTY, **done), step_id).done is True


def test_the_classes_step_is_not_done_while_a_detector_class_is_missing() -> None:
    facts = dataclasses.replace(DONE, active_classes=6, detector_classes_missing=29)
    step = _step(facts, 'classes')
    assert step.done is False
    assert step.action == 'adopt_detector_classes'
    assert step.detail == '29 detector classes are not in the registry.'
    assert build_steps(facts).next == step


def test_the_classes_step_offers_adopt_for_items_the_detector_named_without_a_class() -> None:
    facts = dataclasses.replace(DONE, detector_items_undecided=1)
    step = _step(facts, 'classes')
    assert (step.done, step.action) == (False, 'adopt_detector_classes')
    assert step.detail == '1 item has a detector class and no class.'
    both = dataclasses.replace(facts, detector_classes_missing=2)
    assert _step(both, 'classes').detail == (
        '2 detector classes are not in the registry; 1 item has a detector class and no class.'
    )


def test_the_classes_step_is_done_and_offers_nothing_once_the_detector_is_adopted() -> None:
    step = _step(DONE, 'classes')
    assert (step.done, step.action) == (True, None)
    assert step.detail == '2 classes in the registry.'
    # An empty registry is still not done, with nothing to adopt.
    empty = _step(dataclasses.replace(DONE, active_classes=0), 'classes')
    assert (empty.done, empty.action) == (False, None)


def test_a_step_says_what_is_missing_in_a_sentence() -> None:
    assert _step(
        dataclasses.replace(DONE, hotkeys_missing=2, hotkeys_free=3), 'hotkeys'
    ).detail == ('2 classes you created have no hotkey letter; 3 letters are free.')
    assert _step(EMPTY, 'ingest').detail == 'No images yet.'


def test_review_disagreements_appears_only_after_a_model_is_promoted() -> None:
    assert 'review_disagreements' not in [
        s.id for s in build_steps(dataclasses.replace(DONE, promoted_model=False)).steps
    ]
    assert _step(dataclasses.replace(DONE, disagreements=1), 'review_disagreements').done is False


def test_hotkeys_step_is_not_done_on_a_project_with_no_classes() -> None:
    step = _step(EMPTY, 'hotkeys')
    assert step.done is False
    assert step.detail == 'There are no classes to give a letter to.'


def test_train_step_needs_a_finished_run_that_covers_the_trainable_classes() -> None:
    started = dataclasses.replace(EMPTY, training_runs=2)
    step = _step(started, 'train')
    assert step.done is False
    assert step.detail == '2 training runs started, none finished.'
    partial = dataclasses.replace(DONE, training_classes_missing=3)
    step = _step(partial, 'train')
    assert step.done is False
    assert 'left out 3 classes' in step.detail
    assert _step(DONE, 'train').done is True


def test_after_a_promote_the_probe_comes_before_the_disagreements_queue() -> None:
    unscored = dataclasses.replace(DONE, probe_scored=False)
    ids = [s.id for s in build_steps(unscored).steps]
    assert 'probe' in ids
    assert 'review_disagreements' not in ids
    assert _step(unscored, 'probe').section == 'train'
    assert 'Find disagreements with the new model' in _step(unscored, 'probe').detail
    following = build_steps(unscored).next
    assert following is not None
    assert following.id == 'probe'


def test_disagreements_step_carries_the_queue_count_and_is_done_when_empty() -> None:
    open_ = dataclasses.replace(DONE, disagreements=4)
    step = _step(open_, 'review_disagreements')
    assert (step.done, step.detail) == (False, '4 items in Disagreements.')
    assert _step(DONE, 'review_disagreements').done is True
    assert 'probe' not in [s.id for s in build_steps(DONE).steps]


def test_everything_done_has_no_next_step() -> None:
    result = build_steps(dataclasses.replace(DONE, promoted_model=False))
    assert result.next is None
    assert build_steps(DONE).next is None


def test_default_view_is_gallery_without_a_serving_partition() -> None:
    facts = dataclasses.replace(DONE, item_partition_serving=False)
    assert build_steps(facts).workspace_default_view == 'gallery'


def test_default_view_is_clusters_with_one() -> None:
    assert build_steps(DONE).workspace_default_view == 'clusters'


def test_every_served_step_link_is_a_page_the_app_has() -> None:
    """The app has no near-duplicates page and no bare audit page: a step links to the Workspace
    Duplicates view and to Data quality's accuracy audit."""
    facts = dataclasses.replace(
        DONE, image_near_dup_run=False, audit_pending=False, promoted_model=True
    )
    steps = {s.id: s for s in build_steps(facts).steps}
    assert (steps['near_dups'].section, steps['near_dups'].query) == (
        'workspace',
        'view=gallery&near_dup=collapsed',
    )
    assert steps['audit'].section == 'quality/audit'
    pages = {'ingest', 'classes', 'workspace', 'quality/audit', 'export', 'train'}
    assert {s.section for s in steps.values()} <= pages


def test_the_hotkey_step_is_done_when_no_free_letter_remains() -> None:
    stuck = dataclasses.replace(DONE, hotkeys_missing=4, hotkeys_free=0)
    step = _step(stuck, 'hotkeys')
    assert step.done is True
    assert step.detail == 'Every free hotkey letter is in use.'


def test_class_facts_count_only_user_created_classes_as_missing_a_letter() -> None:
    from types import SimpleNamespace

    from src.services.curation.class_counts import ClassCounts
    from src.services.curation.workflow_steps import class_facts

    def cls(i: int, group: str, letter: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            class_id=i, group=group, hotkey_letter=letter, deprecated=False, merged_into=None
        )

    classes = [cls(0, 'detector'), cls(1, 'region'), cls(2, 'unknown'), cls(3, 'unknown', 'a')]
    active, missing, _below, free = class_facts(classes, ClassCounts(), reserved=['b', 'c'])
    assert (active, missing, free) == (4, 1, 23)
