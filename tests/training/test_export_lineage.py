"""The lineage a run records about its export and about the API build (no database)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.services.curation.export_support import frozen_test_sha_of, label_content_sha
from src.services.training import lineage
from src.services.training.job_models import TrainJobSpec


if TYPE_CHECKING:
    from pathlib import Path


def _export(tmp_path: Path, manifest: dict | None) -> Path:
    root = tmp_path / 'export'
    (root / 'labels' / 'test').mkdir(parents=True)
    (root / 'labels' / 'test' / 'a.txt').write_text('0 0.5 0.5 0.1 0.1\n')
    if manifest is not None:
        (root / 'manifest.json').write_text(json.dumps(manifest))
    return root


def test_the_identity_is_copied_from_a_full_manifest(tmp_path: Path) -> None:
    root = _export(
        tmp_path,
        {
            'dataset_sha': 'dataset-sha-1',
            'frozen_test_sha': 'export-sha-1',
            'test_label_sha': 'label-sha-1',
            'version_tag': 'v1',
        },
    )
    assert lineage.read_export_identity(root) == lineage.ExportIdentity(
        'dataset-sha-1', 'export-sha-1', 'label-sha-1', 'v1'
    )


def test_an_older_manifest_gets_the_test_hashes_computed_from_disk(tmp_path: Path) -> None:
    root = _export(tmp_path, {})
    identity = lineage.read_export_identity(root)
    assert identity.dataset_sha is None  # the export's own claim; never computed
    assert identity.frozen_test_sha == frozen_test_sha_of(root)
    assert identity.test_label_sha == label_content_sha(root, None, truncate=16, split='test')
    assert identity.frozen_test_sha
    assert identity.test_label_sha


def test_no_manifest_and_no_export_leave_the_hashes_absent(tmp_path: Path) -> None:
    identity = lineage.read_export_identity(tmp_path / 'missing')
    assert (identity.dataset_sha, identity.frozen_test_sha, identity.test_label_sha) == (
        None,
        None,
        None,
    )


def test_the_api_sha_is_the_build_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_BUILD_SHA', 'apisha123')
    assert lineage.api_sha() == 'apisha123'


def test_the_start_payload_refuses_a_job_id_and_the_retired_image_fields() -> None:
    for retired in ('job_id', 'trainer_image_id', 'trainer_image_revision'):
        with pytest.raises(ValueError, match='Extra inputs are not permitted'):
            TrainJobSpec(dataset_export_dir='/x', **{retired: 'v'})  # type: ignore[arg-type]
