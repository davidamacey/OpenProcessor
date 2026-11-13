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
    promoted_model=True,
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
        ('hotkeys', {'hotkeys_missing': 2}, {'hotkeys_missing': 0}),
        ('near_dups', {'image_near_dup_run': False}, {'image_near_dup_run': True}),
        ('export', {'exports': 0}, {'exports': 1}),
        ('train', {'training_runs': 0}, {'training_runs': 1}),
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
    assert _step(dataclasses.replace(DONE, hotkeys_missing=2), 'hotkeys').detail == (
        '2 classes have no hotkey letter.'
    )
    assert _step(EMPTY, 'ingest').detail == 'No images yet.'


def test_review_disagreements_appears_only_after_a_model_is_promoted() -> None:
    assert 'review_disagreements' not in [
        s.id for s in build_steps(dataclasses.replace(DONE, promoted_model=False)).steps
    ]
    assert _step(DONE, 'review_disagreements').done is False


def test_everything_done_has_no_next_step() -> None:
    result = build_steps(dataclasses.replace(DONE, promoted_model=False))
    assert result.next is None


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
