"""Live cohort-(e) scenarios: the YOLO export write path.

The exporter is the only write endpoint whose product is files rather than
documents, so these assertions read the bytes it wrote through the harness
bind mount rather than trusting the response envelope.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from .conftest import EXPORTS_DIR


pytestmark = pytest.mark.live

# Container-side export root, as configured in docker/test/compose.yml. The
# API reports paths in its own namespace; the test reads the same bytes
# through the host side of the bind mount.
CONTAINER_EXPORT_ROOT = '/verify-data/exports'


def _host_path(container_path: str) -> Any:
    assert container_path.startswith(CONTAINER_EXPORT_ROOT), container_path
    relative = container_path[len(CONTAINER_EXPORT_ROOT) :].lstrip('/')
    return EXPORTS_DIR / relative if relative else EXPORTS_DIR


@pytest.fixture(scope='module')
def first_export(api_client: Any) -> dict[str, Any]:
    resp = api_client.post('/export/yolo', json={'version_tag': 'live-a', 'seed': 42})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_export_writes_labels_artifacts_and_flips_the_current_symlink(
    first_export: dict[str, Any],
) -> None:
    body = first_export
    assert body['status'] == 'success'
    assert body['version_tag'] == 'live-a'
    counts = body['split_counts']
    total = counts['train'] + counts['val'] + counts['test']
    assert total > 0, body
    assert len(body['dataset_sha']) == 64

    export_dir = _host_path(body['export_dir'])
    assert export_dir.is_dir(), export_dir

    # One YOLO label file per exported item, each a single well-formed row.
    label_files = sorted(export_dir.glob('labels/*/*.txt'))
    assert len(label_files) == total, (len(label_files), counts)
    parts = label_files[0].read_text().split()
    assert len(parts) == 5, parts
    assert parts[0].isdigit()
    assert all(0.0 <= float(p) <= 1.0 for p in parts[1:])

    # Every frozen holdout row must land in the test split, never train/val.
    assert counts['test'] > 0, counts

    for artifact in ('manifest.json', 'data.yaml', 'class_registry.json', 'label_stats.json'):
        assert (export_dir / artifact).is_file(), artifact

    manifest = json.loads((export_dir / 'manifest.json').read_text())
    assert manifest['dataset_sha'] == body['dataset_sha']
    assert manifest['image_count'] == total
    assert manifest['seed'] == 42

    # class_registry.json carries the dense export-id map the training
    # preflight reads back — the artifact that used to have readers and no
    # producer at all.
    registry = json.loads((export_dir / 'class_registry.json').read_text())
    assert registry['export_id_map'], registry
    assert set(registry['export_id_map'].values()) == set(range(len(registry['export_id_map'])))

    current = EXPORTS_DIR / 'current'
    assert current.is_symlink(), 'the current symlink was not flipped'
    # The symlink stores the API container's own absolute path, so compare
    # the link text rather than resolving it on this side of the mount.
    assert current.readlink().as_posix() == body['export_dir']


def test_export_is_deterministic_across_two_tags(
    api_client: Any, first_export: dict[str, Any]
) -> None:
    resp = api_client.post('/export/yolo', json={'version_tag': 'live-b', 'seed': 42})
    assert resp.status_code == 200, resp.text
    second = resp.json()

    assert second['export_dir'] != first_export['export_dir']
    assert second['version_tag'] == 'live-b'
    # Same data + same seed => identical checksum and identical split sizes.
    assert second['dataset_sha'] == first_export['dataset_sha']
    assert second['split_counts'] == first_export['split_counts']

    first_labels = {
        p.relative_to(_host_path(first_export['export_dir']))
        for p in _host_path(first_export['export_dir']).glob('labels/*/*.txt')
    }
    second_labels = {
        p.relative_to(_host_path(second['export_dir']))
        for p in _host_path(second['export_dir']).glob('labels/*/*.txt')
    }
    assert first_labels == second_labels

    assert (EXPORTS_DIR / 'current').readlink().as_posix() == second['export_dir']


def test_export_status_and_dataset_listing_see_both_runs(api_client: Any) -> None:
    status = api_client.get('/export/status')
    assert status.status_code == 200, status.text
    assert status.json()['status'] == 'success'

    listing = api_client.get('/export/datasets')
    assert listing.status_code == 200, listing.text
    datasets = listing.json()['datasets']
    tags = {d['version_tag'] for d in datasets}
    assert {'live-a', 'live-b'} <= tags, tags
    assert sum(1 for d in datasets if d['is_current']) == 1


@pytest.mark.parametrize(
    ('artifact', 'content_type'),
    [
        ('class_registry.json', 'application/json'),
        ('data.yaml', 'application/x-yaml'),
        ('manifest.json', 'application/json'),
        ('label_stats.json', 'application/json'),
    ],
)
def test_registry_artifact_is_served_byte_for_byte(
    api_client: Any, artifact: str, content_type: str
) -> None:
    current_dir = api_client.get('/export/status').json()['export_dir']
    resp = api_client.get(f'/export/registry/{artifact}')
    assert resp.status_code == 200, resp.text
    assert resp.headers['content-type'].startswith(content_type)
    assert resp.headers['cache-control'] == 'no-store'

    on_disk = (_host_path(current_dir) / artifact).read_bytes()
    assert resp.content == on_disk


def test_unknown_export_artifact_is_a_404(api_client: Any) -> None:
    for artifact in ('secrets.json', '../../etc/passwd'):
        resp = api_client.get(f'/export/registry/{artifact}')
        assert resp.status_code == 404, (artifact, resp.status_code, resp.text)
