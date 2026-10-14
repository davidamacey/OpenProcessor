"""GET /bakeoff/eval_datasets and /bakeoff/trained_models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from curation.bakeoff_router._helpers import LIVE_REMAP, _live_export, make_export, make_run
from scripts.curation.bakeoff.freeze import test_sha as freeze_test_sha
from src.services.curation.export_support import frozen_test_sha_of


if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from fastapi.testclient import TestClient


# =============================================================================
# GET /bakeoff/eval_datasets
# =============================================================================


def test_eval_datasets_lists_multiclass_export(env: dict[str, Path], client: TestClient) -> None:
    d = _live_export(env, background=('bg_0',))
    body = client.get(
        '/curation/projects/default/bakeoff/eval_datasets', params={'source': 'export'}
    ).json()
    assert body['count'] == 1
    [row] = body['datasets']
    assert row['id'] == 'export:20260924T233203Z'
    assert (row['source'], row['group'], row['name']) == ('export', None, '20260924T233203Z')
    assert row['path'] == str(d)
    assert row['dataset_kind'] == 'multi_class'
    assert row['nc'] == 84
    assert [
        (c['eval_class_id'], c['name'], c['n_objects'], c['n_images']) for c in row['classes']
    ] == [
        (37, 'miata', 5, 5),
        (38, 'minicooper', 5, 5),
        (43, 'mustang', 5, 5),
        (51, 'porsche', 5, 5),
        (78, 'vw', 5, 5),
    ]
    assert row['classes'][0]['registry_class_id'] == 38
    assert (row['n_images'], row['n_objects'], row['n_background_images']) == (26, 25, 1)
    assert row['frozen_test_sha'] == frozen_test_sha_of(d)
    assert row['test_label_sha'] == freeze_test_sha(d)[0]
    assert row['sha_source'] == 'computed'
    assert row['dataset_sha'] == 'd' * 64
    assert row['unlabeled_items_on_exported_images'] == 69
    assert row['exported_at'] == '2026-09-24T23:32:11+00:00'
    assert row['frozen_ok'] is None
    assert row['is_current'] is False


def test_eval_datasets_registry_class_name_resolved_by_id_with_gaps(
    env: dict[str, Path], client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``registry_class_name`` is looked up by ``registry_class_id`` -- never
    assumed equal to the eval dataset's own class name -- and stays null
    for a registry id that has a gap (deprecated/removed) at that slot."""
    from src.services.curation import eval_datasets

    class _Cls:
        def __init__(self, class_id: int, class_name: str) -> None:
            self.class_id = class_id
            self.class_name = class_name

    class _Loaded:
        def __init__(self, classes: list[_Cls]) -> None:
            self.classes = classes

    class _FakeRegistry:
        def load(self) -> _Loaded:
            # LIVE_EXPORT_ID_MAP maps registry r -> export r-1, so the
            # export's 5 test classes (eval ids 37/38/43/51/78) carry
            # registry ids 38/39/44/52/79. Registry id 38 is gone here --
            # a gap in the id space -- while the other four are present,
            # renamed (so the test can't pass by coincidentally reusing the
            # eval dataset's own class name).
            return _Loaded(
                [
                    _Cls(39, 'miata-renamed'),
                    _Cls(44, 'mustang-renamed'),
                    _Cls(52, 'porsche-renamed'),
                    _Cls(79, 'vw-renamed'),
                ]
            )

    monkeypatch.setattr(eval_datasets, 'get_class_registry', _FakeRegistry)

    _live_export(env, background=('bg_0',))
    body = client.get(
        '/curation/projects/default/bakeoff/eval_datasets', params={'source': 'export'}
    ).json()
    [row] = body['datasets']
    by_registry_id = {c['registry_class_id']: c['registry_class_name'] for c in row['classes']}
    assert by_registry_id == {
        38: None,  # gap: registry entry no longer exists
        39: 'miata-renamed',
        44: 'mustang-renamed',
        52: 'porsche-renamed',
        79: 'vw-renamed',
    }


def test_eval_datasets_sha_source_manifest(env: dict[str, Path], client: TestClient) -> None:
    make_export(
        env['exports'],
        'e1',
        nc=2,
        test_labels={'a': [0], 'b': [1]},
        manifest={'frozen_test_sha': 'f' * 16, 'test_label_sha': 'a' * 16},
    )
    [row] = client.get('/curation/projects/default/bakeoff/eval_datasets').json()['datasets']
    assert (row['frozen_test_sha'], row['test_label_sha'], row['sha_source']) == (
        'f' * 16,
        'a' * 16,
        'manifest',
    )


def test_eval_datasets_lists_single_class_export_at_depth_two(
    env: dict[str, Path], client: TestClient
) -> None:
    profile_dir = env['exports'] / 'region_widget'
    make_export(
        env['exports'],
        'region_widget/20260924T032032Z',
        nc=1,
        names=['widget'],
        test_labels={'a': [0], 'b': [0, 0]},
        manifest={'dataset_kind': 'single_class', 'exported_at': '2026-09-24T03:20:32+00:00'},
    )
    (profile_dir / 'current').symlink_to(profile_dir / '20260924T032032Z')
    # A symlinked top-level "current" and a non-export dir are not listed.
    (env['exports'] / 'current').symlink_to(profile_dir / '20260924T032032Z')
    (env['exports'] / 'scratch').mkdir()
    rows = client.get('/curation/projects/default/bakeoff/eval_datasets').json()['datasets']
    assert [r['id'] for r in rows] == ['export:region_widget/20260924T032032Z']
    [row] = rows
    assert row['is_current'] is True
    assert row['dataset_kind'] == 'single_class'
    assert (row['nc'], row['n_objects'], row['classes'][0]['registry_class_id']) == (1, 3, None)


def test_eval_datasets_external_kept_optional(env: dict[str, Path], client: TestClient) -> None:
    from scripts.curation.bakeoff.freeze import freeze

    good = make_export(env['external'], 'curated/set_a', nc=1, test_labels={'a': [0]})
    freeze(good)
    drifted = make_export(env['external'], 'public/set_b', nc=1, test_labels={'b': [0]})
    freeze(drifted)
    lbl = drifted / 'labels' / 'test' / 'b.txt'
    lbl.chmod(0o644)
    lbl.write_text('0 0.1 0.1 0.1 0.1\n')
    make_export(env['external'], 'sample/unfrozen', nc=1, test_labels={'c': [0]})

    body = client.get(
        '/curation/projects/default/bakeoff/eval_datasets', params={'source': 'external'}
    ).json()
    by_id = {r['id']: r for r in body['datasets']}
    assert set(by_id) == {'external:curated/set_a', 'external:public/set_b'}
    assert by_id['external:curated/set_a']['group'] == 'curated'
    assert by_id['external:curated/set_a']['source'] == 'external'
    assert by_id['external:curated/set_a']['dataset_kind'] == 'external'
    assert by_id['external:curated/set_a']['frozen_ok'] is True
    assert by_id['external:public/set_b']['frozen_ok'] is False
    assert by_id['external:curated/set_a']['dataset_sha'] is None
    # Nothing under the export root: the export listing is empty, not an error.
    assert client.get(
        '/curation/projects/default/bakeoff/eval_datasets', params={'source': 'export'}
    ).json() == {
        'datasets': [],
        'count': 0,
    }


# =============================================================================


# =============================================================================
# GET /bakeoff/trained_models
# =============================================================================


def test_trained_models_for_dataset(env: dict[str, Path], client: TestClient) -> None:
    d = _live_export(env, train_labels={'img_37_0': [37], 'train_only': [38]})
    make_run(
        env,
        'run5',
        export_dir=d,
        class_remap=LIVE_REMAP,
        frozen_test_sha=frozen_test_sha_of(d),
        eval_block={'map50': 0.9356, 'split': 'test'},
    )
    make_run(env, 'unfinished', export_dir=d, state='running')
    body = client.get(
        '/curation/projects/default/bakeoff/trained_models',
        params={'dataset_id': 'export:20260924T233203Z'},
    ).json()
    assert body['count'] == 1
    [m] = body['models']
    assert m['run_id'] == 'run5'
    assert (m['model_family'], m['model_size'], m['imgsz']) == ('yolo26', 'n', 640)
    assert m['train_export_id'] == 'export:20260924T233203Z'
    assert m['class_names'] == ['miata', 'minicooper', 'mustang', 'porsche', 'vw']
    assert m['single_cls'] is False
    assert (m['trainer_map50'], m['trainer_map50_split']) == (0.9356, 'test')
    fd = m['for_dataset']
    assert fd['dataset_id'] == 'export:20260924T233203Z'
    assert fd['same_export'] is True
    assert fd['same_frozen_test'] is True
    assert fd['n_classes_mapped'] == 5
    assert fd['train_test_overlap'] == {'n_images': 1, 'fraction': 1 / 25}

    plain = client.get('/curation/projects/default/bakeoff/trained_models').json()['models'][0]
    assert plain['for_dataset'] is None


# =============================================================================
