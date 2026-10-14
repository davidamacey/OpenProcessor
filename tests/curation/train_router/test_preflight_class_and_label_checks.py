"""Preflight include_classes resolution, empty-label scans and dropped-class warnings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient


# =============================================================================
# include_classes resolvable against the export (P2-8)
# =============================================================================


def test_unresolvable_include_classes_flags_unknown_ids(tmp_path: Any) -> None:
    import json

    from src.services.training.preflight_checks import unresolvable_include_classes

    export_dir = tmp_path / 'export'
    export_dir.mkdir()
    (export_dir / 'class_registry.json').write_text(json.dumps({'export_id_map': {'1': 0, '2': 1}}))

    assert unresolvable_include_classes(str(export_dir), [1, 2]) == []
    assert unresolvable_include_classes(str(export_dir), [1, 999]) == [999]


def test_unresolvable_include_classes_no_export_id_map_flags_all(tmp_path: Any) -> None:
    """A pre-Phase-5 export (no export_id_map) can't resolve any id — every
    requested class is unresolvable, not a silent pass."""
    import json

    from src.services.training.preflight_checks import unresolvable_include_classes

    export_dir = tmp_path / 'export_old'
    export_dir.mkdir()
    (export_dir / 'class_registry.json').write_text(json.dumps({'classes': []}))

    assert unresolvable_include_classes(str(export_dir), [1, 2]) == [1, 2]


def test_preflight_blocks_unresolvable_include_classes(
    app_client: TestClient, tmp_path: Any, project_export_root: Path
) -> None:
    import json

    export_dir = project_export_root / 'export'
    export_dir.mkdir()
    (export_dir / 'class_registry.json').write_text(json.dumps({'export_id_map': {'1': 0}}))

    body = {
        'dataset_export_dir': str(export_dir),
        'profile': 'medium',
        'include_classes': [1, 999],
    }
    r = app_client.post('/curation/projects/default/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    check = next(c for c in out['checks'] if c['name'] == 'include_classes_resolvable')
    assert check['severity'] == 'block'
    assert 999 in check['detail']['unresolvable_class_ids']
    assert out['blocked'] is True


def test_preflight_lpr_single_class_alias_is_retired(
    app_client: TestClient, tmp_path: Any, project_export_root: Path
) -> None:
    """The accepted alias 'lpr_single_class' for dataset_kind is
    dropped; only 'single_class' is recognized. A manifest carrying the
    retired alias must be treated as an ordinary multi-class export, so
    the include_classes-resolvable check runs (and blocks on an
    unresolvable class id) instead of being skipped."""
    import json

    export_dir = project_export_root / 'export_lpr_legacy'
    export_dir.mkdir()
    (export_dir / 'manifest.json').write_text(
        json.dumps(
            {
                'dataset_kind': 'lpr_single_class',
                'positive_images': 1000,
                'split_counts': {'test': 100},
            }
        )
    )

    body = {
        'dataset_export_dir': str(export_dir),
        'profile': 'medium',
        'include_classes': [999],
    }
    r = app_client.post('/curation/projects/default/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    check = next(c for c in out['checks'] if c['name'] == 'include_classes_resolvable')
    assert check['severity'] == 'block'
    assert 999 in check['detail']['unresolvable_class_ids']


def test_preflight_skips_include_classes_check_for_single_class(
    app_client: TestClient, tmp_path: Any, project_export_root: Path
) -> None:
    """Single-class jobs have no include_classes concept — the check must
    not even appear, per the existing dataset_kind branch."""
    import json

    export_dir = project_export_root / 'export_single_class_no_include'
    export_dir.mkdir()
    (export_dir / 'manifest.json').write_text(
        json.dumps(
            {
                'dataset_kind': 'single_class',
                'positive_images': 1000,
                'split_counts': {'test': 100},
            }
        )
    )

    body = {
        'dataset_export_dir': str(export_dir),
        'profile': 'medium',
        'include_classes': [999],  # would be unresolvable for a vehicle export
    }
    r = app_client.post('/curation/projects/default/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    names = {c['name'] for c in out['checks']}
    assert 'include_classes_resolvable' not in names


def test_preflight_accepts_the_generic_single_class_dataset_kind(
    app_client: TestClient, tmp_path: Any, project_export_root: Path
) -> None:
    """A dataset built by ``POST /curation/export/single_class`` writes
    ``dataset_kind='single_class'``; preflight must take the manifest-driven
    branch for it, and name the target class from the manifest rather than
    a hardcoded vocabulary."""
    import json

    export_dir = project_export_root / 'export_single_class'
    export_dir.mkdir()
    (export_dir / 'manifest.json').write_text(
        json.dumps(
            {
                'dataset_kind': 'single_class',
                'class_name': 'street_sign',
                'positive_images': 1200,
                'split_counts': {'train': 1000, 'val': 100, 'test': 100},
            }
        )
    )

    body = {'dataset_export_dir': str(export_dir), 'profile': 'medium'}
    r = app_client.post('/curation/projects/default/train/preflight', json=body)

    assert r.status_code == 200, r.text
    out = r.json()
    balance = next(c for c in out['checks'] if c['name'] == 'class_balance')
    assert balance['severity'] == 'ok'
    assert 'street_sign' in balance['message']
    assert 'license_plate' not in balance['message']
    # The manifest-driven branch also skips the label scan entirely.
    assert next(c for c in out['checks'] if c['name'] == 'empty_labels')['severity'] == 'ok'


# =============================================================================
# empty_labels / region_pairing real scan (P2-8)
# =============================================================================


def test_preflight_empty_labels_blocks_when_whole_export_is_empty(
    app_client: TestClient, tmp_path: Any, project_export_root: Path
) -> None:
    import json

    export_dir = project_export_root / 'export'
    (export_dir / 'labels' / 'train').mkdir(parents=True)
    (export_dir / 'labels' / 'train' / 'a.txt').write_text('')
    (export_dir / 'class_registry.json').write_text(
        json.dumps({'classes': [{'class_id': 1, 'class_name': 'sedan'}], 'export_id_map': {'1': 0}})
    )
    (export_dir / 'manifest.json').write_text(json.dumps({'dataset_kind': 'vehicle'}))

    body = {'dataset_export_dir': str(export_dir), 'profile': 'medium'}
    r = app_client.post('/curation/projects/default/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    check = next(c for c in out['checks'] if c['name'] == 'empty_labels')
    assert check['severity'] == 'block'
    assert out['blocked'] is True


def test_preflight_single_class_export_skips_scan_and_reports_not_applicable(
    app_client: TestClient, tmp_path: Any, project_export_root: Path
) -> None:
    import json

    export_dir = project_export_root / 'export_single_class_scan_skip'
    export_dir.mkdir()
    (export_dir / 'manifest.json').write_text(
        json.dumps(
            {
                'dataset_kind': 'single_class',
                'positive_images': 900,
                'total_images': 1000,
                'split_counts': {'test': 100},
            }
        )
    )

    body = {'dataset_export_dir': str(export_dir), 'profile': 'medium'}
    r = app_client.post('/curation/projects/default/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    empty_check = next(c for c in out['checks'] if c['name'] == 'empty_labels')
    region_check = next(c for c in out['checks'] if c['name'] == 'region_pairing')
    assert empty_check['severity'] == 'ok'
    assert 'background' in empty_check['message'].lower()
    assert region_check['severity'] == 'ok'
    assert 'not applicable' in region_check['message'].lower()


def test_preflight_warns_on_export_that_dropped_unregistered_class_ids(
    app_client: TestClient, tmp_path: Any, project_export_root: Path
) -> None:
    import json as _json

    export_dir = project_export_root / 'export_with_ghosts'
    export_dir.mkdir()
    (export_dir / 'manifest.json').write_text(
        _json.dumps({'dropped_unregistered_class_ids': {'10000': 2}})
    )
    r = app_client.post(
        '/curation/projects/default/train/preflight', json={'dataset_export_dir': str(export_dir)}
    )
    assert r.status_code == 200, r.text
    check = next(c for c in r.json()['checks'] if c['name'] == 'unregistered_class_ids')
    assert check['severity'] == 'warn'
    assert check['detail'] == {'dropped_unregistered_class_ids': {'10000': 2}}

    clean = project_export_root / 'clean_export'
    clean.mkdir()
    (clean / 'manifest.json').write_text(_json.dumps({'dropped_unregistered_class_ids': {}}))
    r = app_client.post(
        '/curation/projects/default/train/preflight', json={'dataset_export_dir': str(clean)}
    )
    check = next(c for c in r.json()['checks'] if c['name'] == 'unregistered_class_ids')
    assert check['severity'] == 'ok'
