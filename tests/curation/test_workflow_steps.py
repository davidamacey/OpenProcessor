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
