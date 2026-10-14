"""Preflight never turns an outage or a corrupt file into a clean answer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from src.services.training.preflight_checks import (
    ExportManifestUnreadableError,
    count_pending_ingest,
    ingest_idle_check,
    read_export_manifest,
)


if TYPE_CHECKING:
    from pathlib import Path


class _Down:
    async def count(self, **_: Any) -> dict[str, int]:
        raise RuntimeError('opensearch is down')


@pytest.mark.asyncio
async def test_pending_ingest_is_unknown_when_it_cannot_be_counted() -> None:
    assert await count_pending_ingest(_Down()) is None


def test_an_uncountable_backlog_warns_instead_of_claiming_idle() -> None:
    check = ingest_idle_check(None, ['curation-detection-worker'])
    assert check.severity == 'warn'
    assert 'could not be counted' in check.message


def test_a_counted_backlog_keeps_its_two_outcomes() -> None:
    assert ingest_idle_check(0, ['w']).severity == 'ok'
    busy = ingest_idle_check(7, ['w'])
    assert busy.severity == 'warn'
    assert busy.detail['pending_ingest'] == 7


def test_a_missing_manifest_is_empty(tmp_path: Path) -> None:
    assert read_export_manifest(str(tmp_path)) == {}
    assert read_export_manifest(None) == {}


def test_an_unreadable_manifest_raises(tmp_path: Path) -> None:
    (tmp_path / 'manifest.json').write_text('{not json')
    with pytest.raises(ExportManifestUnreadableError):
        read_export_manifest(str(tmp_path))


def test_a_non_object_manifest_raises(tmp_path: Path) -> None:
    (tmp_path / 'manifest.json').write_text('[1, 2]')
    with pytest.raises(ExportManifestUnreadableError):
        read_export_manifest(str(tmp_path))


def test_a_valid_manifest_is_returned(tmp_path: Path) -> None:
    (tmp_path / 'manifest.json').write_text('{"dataset_kind": "x"}')
    assert read_export_manifest(str(tmp_path)) == {'dataset_kind': 'x'}
