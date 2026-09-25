"""Parity between the API's test-split identity helpers and the trainer's
stdlib-only ``freeze.test_sha``.

``label_content_sha(..., split='test')`` (API side, imports ``src``) and
``freeze.test_sha`` (trainer/evaluator side, stdlib-only, no ``src`` import)
must agree byte-for-byte on the same directory, or a run frozen by one and
verified by the other would silently drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.services.curation.export_support import frozen_test_sha_of, label_content_sha


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.curation.bakeoff import freeze  # noqa: E402


def _write_export(root: Path) -> None:
    for split in ('train', 'val', 'test'):
        (root / 'labels' / split).mkdir(parents=True)
    (root / 'labels' / 'test' / 'a.txt').write_text('0 0.5 0.5 0.1 0.1\n')
    (root / 'labels' / 'test' / 'nested').mkdir()
    (root / 'labels' / 'test' / 'nested' / 'b.txt').write_text('')  # empty label file
    (root / 'labels' / 'train' / 'c.txt').write_text('1 0.2 0.2 0.1 0.1\n')


def test_label_content_sha_matches_freeze_test_sha_on_a_populated_split(tmp_path):
    _write_export(tmp_path)
    api_sha = label_content_sha(tmp_path, None, truncate=16, split='test')
    trainer_sha, n = freeze.test_sha(tmp_path)
    assert api_sha == trainer_sha
    assert n == 2


def test_label_content_sha_empty_string_with_no_test_split(tmp_path):
    """A missing ``labels/test/`` is the exporter's own "never happened"
    marker (``''``). ``freeze.test_sha`` has no such special case -- it
    still hashes zero bytes -- so this is the one input where the two
    helpers deliberately diverge; parity is asserted only when a test
    split actually exists (the case that matters for a real export)."""
    (tmp_path / 'labels' / 'train').mkdir(parents=True)
    (tmp_path / 'labels' / 'train' / 'c.txt').write_text('0 0.1 0.1 0.1 0.1\n')
    assert label_content_sha(tmp_path, None, truncate=16, split='test') == ''
    _, n = freeze.test_sha(tmp_path)
    assert n == 0


def test_frozen_test_sha_of_unchanged_by_a_label_content_edit(tmp_path):
    _write_export(tmp_path)
    before = frozen_test_sha_of(tmp_path)
    (tmp_path / 'labels' / 'test' / 'a.txt').write_text('0 0.9 0.9 0.9 0.9\n')
    assert frozen_test_sha_of(tmp_path) == before


def test_frozen_test_sha_of_changed_by_adding_a_test_file(tmp_path):
    _write_export(tmp_path)
    before = frozen_test_sha_of(tmp_path)
    (tmp_path / 'labels' / 'test' / 'new.txt').write_text('0 0.1 0.1 0.1 0.1\n')
    assert frozen_test_sha_of(tmp_path) != before


def test_test_label_sha_changed_by_a_content_edit(tmp_path):
    _write_export(tmp_path)
    before = label_content_sha(tmp_path, None, truncate=16, split='test')
    (tmp_path / 'labels' / 'test' / 'a.txt').write_text('0 0.9 0.9 0.9 0.9\n')
    after = label_content_sha(tmp_path, None, truncate=16, split='test')
    assert before != after


def test_freeze_verify_accepts_a_lock_with_only_the_legacy_key(tmp_path):
    _write_export(tmp_path)
    sha, n = freeze.test_sha(tmp_path)
    import json

    (tmp_path / freeze.LOCK_NAME).write_text(
        json.dumps({'frozen_test_sha': sha, 'n_label_files': n, 'split': 'test'})
    )
    ok, msg = freeze.verify(tmp_path)
    assert ok, msg
