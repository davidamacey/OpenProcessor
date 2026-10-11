"""The capture file format: fixed schema, round trip, integrity checks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from scripts.bench import parity_store as ps


if TYPE_CHECKING:
    from pathlib import Path


def test_round_trip_preserves_arrays_and_meta(make_capture, tmp_path: Path) -> None:
    cap = make_capture('ref', engine='fp32')
    ps.save_capture(tmp_path / 'ref', cap)
    back = ps.load_capture(tmp_path / 'ref')
    assert back.meta['tag'] == 'ref'
    assert set(back.arrays) == set(ps.ARRAY_SPEC)
    for name, arr in cap.arrays.items():
        assert back.arrays[name].dtype == arr.dtype
        assert np.array_equal(back.arrays[name], arr)


def test_capture_json_records_schema_and_npz_digest(make_capture, tmp_path: Path) -> None:
    ps.save_capture(tmp_path / 'c', make_capture())
    doc = json.loads((tmp_path / 'c' / 'capture.json').read_text())
    assert doc['schema'] == ps.SCHEMA
    assert doc['npz_sha256'] == ps.sha256_file(tmp_path / 'c' / 'arrays.npz')
    assert doc['arrays']['item_vec']['dtype'] == 'float16'
    assert doc['counts']['items'] == 180


def test_vectors_are_stored_as_float16(make_capture) -> None:
    cap = make_capture()
    assert cap.arrays['item_vec'].dtype == np.float16
    assert cap.arrays['item_box'].dtype == np.float32


def test_load_rejects_a_tampered_npz(make_capture, tmp_path: Path) -> None:
    ps.save_capture(tmp_path / 'c', make_capture())
    with (tmp_path / 'c' / 'arrays.npz').open('ab') as fh:
        fh.write(b'x')
    with pytest.raises(ps.CaptureError, match='sha256'):
        ps.load_capture(tmp_path / 'c')


def test_load_rejects_another_schema(make_capture, tmp_path: Path) -> None:
    ps.save_capture(tmp_path / 'c', make_capture())
    path = tmp_path / 'c' / 'capture.json'
    doc = json.loads(path.read_text())
    doc['schema'] = 99
    path.write_text(json.dumps(doc))
    with pytest.raises(ps.CaptureError, match='schema'):
        ps.load_capture(tmp_path / 'c')


def test_save_rejects_a_wrong_dtype(make_capture, tmp_path: Path) -> None:
    cap = make_capture(mutate=lambda a: a.update(item_vec=a['item_vec'].astype(np.float32)))
    with pytest.raises(ps.CaptureError, match='item_vec'):
        ps.save_capture(tmp_path / 'c', cap)


def test_save_rejects_ragged_groups(make_capture, tmp_path: Path) -> None:
    cap = make_capture(mutate=lambda a: a.update(item_score=a['item_score'][:-1]))
    with pytest.raises(ps.CaptureError, match='item'):
        ps.save_capture(tmp_path / 'c', cap)


def test_save_rejects_missing_meta(make_capture, tmp_path: Path) -> None:
    cap = make_capture()
    del cap.meta['manifest']
    with pytest.raises(ps.CaptureError, match='manifest'):
        ps.save_capture(tmp_path / 'c', cap)


def test_empty_arrays_are_a_valid_capture(tmp_path: Path) -> None:
    arrays = ps.empty_arrays(8)
    meta = {
        'tag': 't',
        'created_at': 'now',
        'manifest': {'sha256': 'x'},
        'engine': {},
        'environment': {},
        'sections': [],
    }
    ps.save_capture(tmp_path / 'e', ps.Capture(meta, arrays))
    assert ps.load_capture(tmp_path / 'e').arrays['item_vec'].shape == (0, 8)
