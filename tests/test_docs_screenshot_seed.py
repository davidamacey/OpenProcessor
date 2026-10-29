"""The docs screenshot seeding (issue #335): the pure decisions the seeder makes before it
touches a stack, and the illustrative-results and stand-in segmenter helpers it relies on."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'frontend' / 'scripts'))
sys.path.insert(0, str(ROOT / 'docker' / 'test'))

import fake_segmenter  # noqa: E402  # path set above
from docs_screenshots import ctx, registry, seed  # noqa: E402  # path set above

from scripts.curation import seed_docs_run_results as results  # noqa: E402  # path set above


def _listing(*slugs: str) -> dict:
    return {'projects': [{'slug': s} for s in ('default', *slugs)]}


def test_an_empty_stack_is_seeded() -> None:
    assert seed.stack_state(_listing()) == 'empty'


def test_a_fully_seeded_stack_is_left_alone() -> None:
    assert seed.stack_state(_listing(*ctx.SEEDED_PROJECTS)) == 'seeded'


@pytest.mark.parametrize('extra', [('private-one',), ('private-one', *ctx.SEEDED_PROJECTS)])
def test_a_stack_holding_anything_else_is_refused(extra: tuple[str, ...]) -> None:
    with pytest.raises(seed.SeedError, match='private-one'):
        seed.stack_state(_listing(*extra))


def test_a_partly_seeded_stack_is_refused() -> None:
    with pytest.raises(seed.SeedError, match=r'beside the seeded ones'):
        seed.stack_state(_listing(ctx.MAIN_PROJECT))


def test_the_import_mapping_goes_by_name_through_the_rename() -> None:
    preview = {
        'classes': [
            {'dataset_class': 'Car'},
            {'dataset_class': 'automobile'},
            {'dataset_class': 'bus'},
        ]
    }
    ids = {'car': 7, 'bus': 9}
    got = seed.import_mapping(preview, ids, seed.YOLO_TO_CLASS)
    assert got == [
        {'dataset_class': 'Car', 'action': 'map', 'class_id': 7},
        {'dataset_class': 'automobile', 'action': 'map', 'class_id': 7},
        {'dataset_class': 'bus', 'action': 'map', 'class_id': 9},
    ]


def test_an_unknown_class_in_the_preview_is_an_error_not_a_guess() -> None:
    with pytest.raises(KeyError):
        seed.import_mapping({'classes': [{'dataset_class': 'zebra'}]}, {'car': 1}, {})


def test_the_wheel_profile_body_is_the_example_without_comment_and_name() -> None:
    example = json.loads(seed.WHEEL_EXAMPLE.read_text())
    body = seed.wheel_profile_body(example)
    assert '_comment' not in body
    assert 'name' not in body
    assert body['segmenter_text_prompt'] == 'wheel'
    assert body['parent_classes'] == ['car']


def test_the_worker_pass_count_is_read_from_its_last_summary_line() -> None:
    log = (
        'x region_worker_done avg_cps=1 elapsed_s=2 processed=155 written=155\n'
        'x region_worker_done avg_cps=0 elapsed_s=1 processed=0 written=0\n'
    )
    assert seed.worker_processed(log) == 0
    assert seed.worker_processed(log.splitlines()[0]) == 155
    assert seed.worker_processed('nothing here') == 0


def test_missing_public_data_is_named(tmp_path: Path) -> None:
    (tmp_path / 'import_fixture').mkdir()
    assert seed.missing_data(tmp_path) == ['coco_va_readme']


def test_staging_refuses_without_the_fetched_data(tmp_path: Path) -> None:
    with pytest.raises(seed.SeedError, match='make sample-coco-import'):
        seed.stage_public_data(tmp_path / 'none', tmp_path / 'source')


def test_staging_copies_each_dataset_once(tmp_path: Path) -> None:
    samples, source = tmp_path / 'samples', tmp_path / 'source'
    for name in seed.STAGED:
        (samples / name).mkdir(parents=True)
        (samples / name / 'a.txt').write_text(name)
    seed.stage_public_data(samples, source)
    (source / 'coco_sample' / 'a.txt').write_text('edited')
    seed.stage_public_data(samples, source)  # a re-run leaves staged data alone
    assert (source / 'coco_sample' / 'a.txt').read_text() == 'edited'
    assert (source / 'import_fixture' / 'a.txt').read_text() == 'import_fixture'


def test_every_seeded_state_is_a_registered_state() -> None:
    assert len(registry.SEEDED_STATES) == 17
    assert set(registry.SEEDED_STATES) <= set(registry.STATES)


def test_the_confusion_matrix_keeps_every_class_support() -> None:
    supports = [50, 30, 20, 7]
    matrix = results.confusion_counts(supports)
    assert [sum(row) for row in matrix] == supports
    assert all(matrix[i][i] > sum(matrix[i]) / 2 for i in range(4))


def test_per_class_rows_are_derived_from_the_matrix() -> None:
    rows = results.per_class_rows(['a', 'b'], [[8, 2], [1, 9]])
    assert rows[0]['support'] == 10
    assert rows[0]['precision'] == round(8 / 9, 3)
    assert rows[1]['recall'] == 0.9


def test_the_stand_in_segmenter_honours_the_request_limits() -> None:
    every = fake_segmenter.segment(
        {'max_candidates': 4, 'return_masks': True, 'text_prompt': 'wheel'}
    )
    assert len(every['candidates']) == 3
    assert all(c['mask_polygon'] for c in every['candidates'])
    capped = fake_segmenter.segment({'max_candidates': 1})
    assert [c['score'] for c in capped['candidates']] == [0.91]
    assert capped['candidates'][0]['mask_polygon'] is None
    floor = fake_segmenter.segment({'max_candidates': 4, 'min_score': 0.5})
    assert [c['score'] for c in floor['candidates']] == [0.91, 0.86]


def test_the_seed_scripts_take_a_project_defaulting_to_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.curation import seed_docs_embeddings as embeddings

    monkeypatch.setenv('OP_CURATION_PROJECT', 'from-env')
    run = ['--jobs-dir', 'j', '--job-id', 'x', '--class', 'a=1']
    assert embeddings.build_parser().parse_args([]).project == 'from-env'
    assert results.build_parser().parse_args(run).project == 'from-env'
    assert results.build_parser().parse_args([*run, '--project', 'p']).project == 'p'


def test_the_embedding_seed_selects_items_through_the_embedding_state_helper() -> None:
    from scripts.curation import seed_docs_embeddings as embeddings
    from src.domain.embedding_state import not_embedded_clause

    assert embeddings.pending_query()['query'] == not_embedded_clause()
