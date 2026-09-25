"""Tests for scripts/datasets/_common.py -- the shared download/hash/sample
helpers both public-sample fetchers (G-05) use. Network is never touched:
``download`` uses ``urllib.request.urlopen`` only when the destination
doesn't already exist, so tests that pre-create files never hit it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.datasets import _common


if TYPE_CHECKING:
    from pathlib import Path


def test_sha256_file_matches_known_value(tmp_path: Path) -> None:
    p = tmp_path / 'f.bin'
    p.write_bytes(b'hello world')
    assert _common.sha256_file(p) == (
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
    )


def test_download_skips_existing_file_with_matching_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / 'img.jpg'
    dest.write_bytes(b'cached bytes')
    expected = _common.sha256_file(dest)

    def _boom(*_a, **_k):
        raise AssertionError('network should not be touched when the file is already correct')

    monkeypatch.setattr(_common.urllib.request, 'urlopen', _boom)
    out = _common.download('http://example.invalid/img.jpg', dest, expected_sha256=expected)
    assert out == dest
    assert dest.read_bytes() == b'cached bytes'


def test_download_redownloads_on_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale on-disk file that doesn't match ``expected_sha256`` must be
    re-fetched, not trusted as-is."""
    dest = tmp_path / 'img.jpg'
    dest.write_bytes(b'stale bytes')
    fresh = b'fresh bytes'
    fresh_sha = _common.hashlib.sha256(fresh).hexdigest()

    class _FakeResp:
        def __init__(self, payload: bytes) -> None:
            self._chunks = [payload, b'']

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, _n):
            return self._chunks.pop(0) if self._chunks else b''

    monkeypatch.setattr(_common.urllib.request, 'urlopen', lambda *_a, **_k: _FakeResp(fresh))
    out = _common.download('http://example.invalid/img.jpg', dest, expected_sha256=fresh_sha)
    assert out.read_bytes() == fresh


def test_seeded_sample_is_deterministic_and_order_independent_of_call_count() -> None:
    pool = list(range(100))
    a = _common.seeded_sample(pool, 10, seed=42)
    b = _common.seeded_sample(pool, 10, seed=42)
    assert a == b
    assert len(a) == 10
    assert len(set(a)) == 10  # no duplicates
    assert all(x in pool for x in a)


def test_seeded_sample_different_seed_differs() -> None:
    pool = list(range(100))
    a = _common.seeded_sample(pool, 10, seed=1)
    b = _common.seeded_sample(pool, 10, seed=2)
    assert a != b


def test_seeded_sample_returns_whole_pool_when_pool_smaller_than_n() -> None:
    pool = [1, 2, 3]
    assert _common.seeded_sample(pool, 10, seed=1) == pool


def test_write_json_and_write_csv_roundtrip(tmp_path: Path) -> None:
    import json

    _common.write_json(tmp_path / 'x.json', {'a': 1})
    assert json.loads((tmp_path / 'x.json').read_text()) == {'a': 1}

    _common.write_csv(tmp_path / 'x.csv', [{'a': '1', 'b': '2'}], ('a', 'b'))
    text = (tmp_path / 'x.csv').read_text()
    assert text.splitlines() == ['a,b', '1,2']


def test_load_pinned_manifest_missing_file_returns_none(tmp_path: Path) -> None:
    assert _common.load_pinned_manifest(tmp_path / 'nope.json') is None


def test_load_pinned_manifest_rejects_non_list(tmp_path: Path) -> None:
    p = tmp_path / 'bad.json'
    p.write_text('{"not": "a list"}', encoding='utf-8')
    with pytest.raises(_common.FetchError):
        _common.load_pinned_manifest(p)


def test_load_pinned_manifest_reads_list(tmp_path: Path) -> None:
    import json

    p = tmp_path / 'ok.json'
    p.write_text(json.dumps([{'image_id': 1}]), encoding='utf-8')
    assert _common.load_pinned_manifest(p) == [{'image_id': 1}]
