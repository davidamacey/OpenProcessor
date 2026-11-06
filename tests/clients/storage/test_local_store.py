"""LocalFsStore: atomic writes, prefix listing and deletion, real paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.clients.storage.base import BlobStore
from src.clients.storage.local import LocalFsStore


UNSAFE = 'unsafe storage|unknown storage root|max_side'


@pytest.fixture
def store(tmp_path: Path) -> LocalFsStore:
    return LocalFsStore(tmp_path)


def test_is_a_blob_store(store: LocalFsStore) -> None:
    assert isinstance(store, BlobStore)


def test_read_after_write(store: LocalFsStore) -> None:
    store.write_atomic('a/b/c.bin', b'hello')
    with store.open_read('a/b/c.bin') as f:
        assert f.read() == b'hello'
    assert store.exists('a/b/c.bin')
    assert not store.exists('a/b/missing.bin')
    assert store.stat('a/b/c.bin').size == 5


def test_crash_mid_write_keeps_the_old_file(
    store: LocalFsStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.write_atomic('k.bin', b'old')

    def boom(self: Path, target: object) -> None:
        raise OSError('simulated crash before the rename')

    monkeypatch.setattr(Path, 'replace', boom)
    with pytest.raises(OSError, match='simulated'):
        store.write_atomic('k.bin', b'new')
    monkeypatch.undo()
    with store.open_read('k.bin') as f:
        assert f.read() == b'old'
    assert [p.name for p in tmp_path.iterdir()] == ['k.bin']


def test_list_prefix_is_segment_based_and_sorted(store: LocalFsStore) -> None:
    for key in ('p/a/2.txt', 'p/a/1.txt', 'p/a/sub/3.txt', 'p/ab/4.txt'):
        store.write_atomic(key, b'x')
    assert store.list_prefix('p/a') == ['p/a/1.txt', 'p/a/2.txt', 'p/a/sub/3.txt']
    assert store.list_prefix('p/nothing') == []


def test_delete_prefix_removes_the_tree_only(store: LocalFsStore) -> None:
    for key in ('p/a/1.txt', 'p/a/sub/2.txt', 'p/ab/3.txt'):
        store.write_atomic(key, b'x')
    assert store.delete_prefix('p/a') == 2
    assert store.list_prefix('p') == ['p/ab/3.txt']
    assert store.delete_prefix('p/a') == 0


def test_local_path_is_the_real_file(store: LocalFsStore, tmp_path: Path) -> None:
    store.write_atomic('x/y.bin', b'1')
    assert store.local_path('x/y.bin') == tmp_path / 'x' / 'y.bin'
    assert store.local_path('x/y.bin').read_bytes() == b'1'


@pytest.mark.parametrize('key', ['../x', '/abs', 'a/../../x', 'a\\b', 'a//b', ''])
def test_unsafe_keys_are_refused(store: LocalFsStore, key: str) -> None:
    with pytest.raises(ValueError, match=UNSAFE):
        store.write_atomic(key, b'x')
    with pytest.raises(ValueError, match=UNSAFE):
        store.exists(key)
