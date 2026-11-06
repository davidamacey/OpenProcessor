"""2E-2: the bundle of the 0.6.0 fixture has the expected counts, and the export only reads."""

from __future__ import annotations

import datetime as dt
import hashlib
from collections import Counter

import pytest

from src.services.upgrade.from_060.bundle import BundleError, check_bundle, iter_records
from src.services.upgrade.from_060.export import export_bundle
from tests.upgrade._v060 import READ_CALLS, V060Install


NOW = dt.datetime(2026, 2, 1, tzinfo=dt.UTC)


async def _export(v060: V060Install, tmp_path):
    return await export_bundle(v060.reader, v060.layout, v060.store, tmp_path / 'bundle', now=NOW)


@pytest.mark.asyncio
async def test_the_fixture_bundle_has_the_expected_counts(v060, tmp_path) -> None:
    manifest = await _export(v060, tmp_path)
    counts = manifest.counts
    assert counts['doc:op_projects'] == 5  # four projects and the revision counter
    assert counts['index:op_projects'] == 1
    assert counts['doc:op_global_configs'] == 6
    # lease and runtime documents are transient: neither exported nor restored
    configs = counts['doc:op_prj_default__configs']
    assert configs == len(
        [
            d
            for d in v060.os.docs('op_prj_default__configs')
            if not d.startswith(('lease:', 'runtime:'))
        ]
    )
    assert 'doc:op_prj_old-demo__configs' not in counts  # a deleted project has no indexes left
    assert counts['file:state:demo'] == 9
    assert counts['file:autolabel:demo'] == 2
    assert counts['file:class_registry:default'] == 1
    assert counts['file:bulk_ingest:demo'] == 4
    assert counts['file:dataset_import:demo'] == 6
    assert counts['blob:train:demo'] == 1  # the run log goes to a storage key
    assert counts['file:models'] == 2


@pytest.mark.asyncio
async def test_the_run_log_is_copied_to_a_storage_key_and_listed(v060, tmp_path) -> None:
    manifest = await _export(v060, tmp_path)
    key = 'projects/demo/training/train_20260104T000000_ab12cd34/run.log'
    assert key in manifest.blobs
    assert v060.store.exists(key)
    (blob,) = [r for r in iter_records(tmp_path / 'bundle') if r['t'] == 'blob']
    assert blob['sha256'] == hashlib.sha256(b'epoch 1/1 ok\n').hexdigest()


@pytest.mark.asyncio
async def test_the_export_issues_reads_only(v060, tmp_path) -> None:
    await _export(v060, tmp_path)
    assert set(v060.os.calls) <= READ_CALLS
    assert Counter(v060.os.calls)['search'] > 0


@pytest.mark.asyncio
async def test_the_export_touches_no_source_file(v060, tmp_path) -> None:
    before = {
        p: (p.stat().st_mtime_ns, p.read_bytes()) for p in v060.root.rglob('*') if p.is_file()
    }
    await _export(v060, tmp_path)
    written = {p for p in v060.root.rglob('*') if p.is_file()} - set(before)
    assert all(p.is_relative_to(v060.root / 'storage') for p in written)  # only the storage copy
    assert before == {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in before}


@pytest.mark.asyncio
async def test_the_bundle_checks_against_its_manifest_and_a_torn_one_is_refused(
    v060, tmp_path
) -> None:
    manifest = await _export(v060, tmp_path)
    assert check_bundle(tmp_path / 'bundle').sha256 == manifest.sha256
    bundle = tmp_path / 'bundle' / 'control_bundle.jsonl.gz'
    bundle.write_bytes(bundle.read_bytes()[:-20])
    with pytest.raises(BundleError):
        check_bundle(tmp_path / 'bundle')


@pytest.mark.asyncio
async def test_the_queue_database_is_kept_as_a_readable_image(v060, tmp_path) -> None:
    await _export(v060, tmp_path)
    (queue,) = [
        r
        for r in iter_records(tmp_path / 'bundle')
        if r['t'] == 'file' and r['rel'].endswith('queue.sqlite')
    ]
    assert queue['bulk'] is True
    assert queue['enc'] == 'b64'
