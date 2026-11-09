"""An uploaded file is stored once under its ``imohash``; a different file that shares the
``imohash`` goes to its own SHA-256 path instead of being served as the first file."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from src.services.curation.image_serving import persist_uploaded_bytes


if TYPE_CHECKING:
    from pathlib import Path

KEY = 'ab' + 'c' * 30


def _config(tmp_path: Path) -> Any:
    return SimpleNamespace(upload_root=tmp_path)  # the only setting the function reads


def test_the_same_bytes_resolve_to_one_file(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    first = persist_uploaded_bytes(b'one', imohash=KEY, extension='.jpg', config=cfg)
    again = persist_uploaded_bytes(b'one', imohash=KEY, extension='.jpg', config=cfg)

    assert first == again == tmp_path / 'ab' / f'{KEY}.jpg'
    assert first.read_bytes() == b'one'


def test_other_bytes_with_the_same_imohash_keep_both_files(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    first = persist_uploaded_bytes(b'one', imohash=KEY, extension='.jpg', config=cfg)
    other = persist_uploaded_bytes(b'two', imohash=KEY, extension='.jpg', config=cfg)

    sha = hashlib.sha256(b'two').hexdigest()
    assert other == tmp_path / sha[:2] / f'{sha}.jpg'
    assert (first.read_bytes(), other.read_bytes()) == (b'one', b'two')
    assert persist_uploaded_bytes(b'two', imohash=KEY, extension='.jpg', config=cfg) == other
