"""``labeled.rows`` serves the detector's unchecked machine labels as their own row, so they are
not counted under "Other"; and the ``label`` workflow step names the queue the work is in."""

from __future__ import annotations

from src.services.curation.stats_dataset import labeled_block, rollup_class_sources
from src.services.curation.workflow_steps import WorkflowFacts, build_steps


def test_the_detector_source_is_its_own_provenance_not_other() -> None:
    rollup = rollup_class_sources(
        [
            {'key': 'detector', 'doc_count': 1131},
            {'key': 'human', 'doc_count': 4},
            {'key': 'mystery_writer', 'doc_count': 2},
        ]
    )
    assert rollup['by_detector'] == 1131
    assert rollup['by_human'] == 4
    assert rollup['other'] == 2


def test_the_rows_name_the_detector_label_and_sum_to_the_total() -> None:
    rollup = rollup_class_sources(
        [{'key': 'detector', 'doc_count': 1131}, {'key': 'external_label', 'doc_count': 185}]
    )
    block = labeled_block(rollup)
    rows = {r['key']: r for r in block['rows']}
    assert rows['by_detector']['label'] == 'Detector, unchecked'
    assert rows['by_detector']['count'] == 1131
    assert rows['other']['count'] == 0
    assert sum(r['count'] for r in block['rows']) == block['total'] == 1316
    assert block['by_detector'] == 1131


def _label_step(*, unchecked: int = 0, below_floor: int = 0) -> str:
    facts = WorkflowFacts(
        images=100,
        active_classes=5,
        machine_labels_unchecked=unchecked,
        classes_below_floor=below_floor,
    )
    return next(s.detail for s in build_steps(facts).steps if s.id == 'label')


def test_the_label_step_names_the_queue_the_machine_labels_are_in() -> None:
    detail = _label_step(unchecked=1131, below_floor=2)
    assert detail.startswith('1,131 machine labels to check in Check machine labels')
    assert '2 classes have too few validated examples' in detail
    steps = build_steps(WorkflowFacts(images=1, active_classes=1, machine_labels_unchecked=1))
    step = next(s for s in steps.steps if s.id == 'label')
    assert step.detail.startswith('1 machine label to check in Check machine labels')
    assert (step.section, step.query) == ('workspace', 'queue=check_machine')


def test_the_label_step_keeps_its_old_copy_when_nothing_waits_in_the_queue() -> None:
    assert _label_step(below_floor=2) == '2 classes have too few validated examples.'
    assert _label_step() == 'Every class has enough validated examples.'
