"""CPU tests of the gpu:PR-0..PR-2 research harness (``scripts/research/gpu_pipeline``).

Only the GPU-free parts: image-set ordering, statistics, the tensor gate, the CPU references, the
set P sampler, the fixture generator and the PR-0/PR-2 tier-0 smoke paths. The GPU runs are
reproduced from the commands in the research note.
"""

from __future__ import annotations

import io
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


_HARNESS = Path(__file__).resolve().parents[2] / 'scripts' / 'research' / 'gpu_pipeline'
sys.path.insert(0, str(_HARNESS))

# ruff: noqa: E402
import cpu_reference as reference
import exif_fixtures
import harness_common as common
import nsys_stats
import pr0_stage_split
import pr2_resize_ops
import set_p_sample


def _photo(path: Path, w: int = 96, h: int = 64, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    img = Image.fromarray(base).resize((w, h), Image.Resampling.BICUBIC)
    img.save(path, 'JPEG', quality=90)
    return path


def test_ordered_files_follow_pin_then_manifest_then_name(tmp_path: Path) -> None:
    for n in ('b.jpg', 'a.jpg', 'c.jpg'):
        _photo(tmp_path / n)
    pin = tmp_path / 'pin.json'
    pin.write_text(json.dumps({'images': [{'file_name': 'c.jpg'}, {'file_name': 'a.jpg'}]}))
    man = tmp_path / 'man.json'
    man.write_text(json.dumps({'files': [{'name': 'b.jpg'}, {'name': 'c.jpg'}]}))
    assert [p.name for p in common.ordered_files(tmp_path, pin=pin)] == ['c.jpg', 'a.jpg']
    assert [p.name for p in common.ordered_files(tmp_path, manifest=man)] == ['b.jpg', 'c.jpg']
    assert [p.name for p in common.ordered_files(tmp_path)] == ['a.jpg', 'b.jpg', 'c.jpg']


def test_take_refuses_a_short_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='registered sample'):
        common.take([tmp_path / 'x.jpg'], 2)


def test_spread_is_max_minus_min_over_median() -> None:
    assert common.spread([9.0, 10.0, 11.0]) == pytest.approx(0.2)


def test_paired_bootstrap_ratio_detects_a_real_gain_and_a_tie() -> None:
    ctrl = [10.0, 11.0, 9.5, 10.5, 10.2, 9.8, 10.1, 10.4, 9.9]
    gain = common.paired_bootstrap_ratio([c * 1.3 for c in ctrl], ctrl)
    tie = common.paired_bootstrap_ratio(ctrl, ctrl)
    assert gain['lo'] > 1.2
    assert tie['ratio'] == pytest.approx(1.0)
    assert tie['lo'] <= 1.0 <= tie['hi']


def test_tensor_gate_passes_small_noise_and_fails_a_shift_or_shape() -> None:
    a = np.full((3, 16, 16), 100, np.uint8)
    assert reference.tensor_gate(a, a).passed
    b = a.copy()
    b[0, 0, 0] = 109  # max abs 9 > 8
    assert not reference.tensor_gate(a, b).passed
    assert not reference.tensor_gate(a, a + 1).passed  # mean abs 1.0 >= 0.5
    assert not reference.tensor_gate(a, a[:, :8]).passed


def test_references_have_the_registered_shapes(tmp_path: Path) -> None:
    frame = reference.pillow_frame(_photo(tmp_path / 'p.jpg', 200, 120))
    assert reference.letterbox_ref(frame).shape == (3, 640, 640)
    assert reference.whole_square_ref(frame).shape == (3, 336, 336)
    assert reference.crop_square_ref(frame, (10, 10, 100, 80)).shape == (3, 336, 336)
    lb = reference.letterbox_ref(frame)
    assert lb[0, 0, 0] == 114  # grey fill above the pasted image


def test_cpu_letterbox_with_the_shipped_weights_passes_the_gate_and_is_deterministic(
    tmp_path: Path,
) -> None:
    frame = reference.pillow_frame(_photo(tmp_path / 'p.jpg', 400, 300, seed=3))
    a, b = pr0_stage_split.cpu_letterbox(frame), pr0_stage_split.cpu_letterbox(frame)
    assert np.array_equal(a, b)
    assert reference.tensor_gate(reference.letterbox_ref(frame), a).passed


def test_tier0_smokes_run_end_to_end(tmp_path: Path) -> None:
    files = [_photo(tmp_path / f'{i}.jpg', seed=i) for i in range(3)]
    s0 = pr0_stage_split.cpu_smoke(files)
    assert s0['items'] == 3
    assert s0['gate_pass'] == 3
    assert s0['deterministic']
    s2 = pr2_resize_ops.cpu_smoke(files)
    assert s2['letterbox_gate_pass'] == 3
    assert s2['shapes_ok']


def test_box_match_uses_one_pixel_tolerance() -> None:
    ref = np.array([[10, 10, 50, 50], [100, 100, 200, 200]], np.float32)
    near = ref + np.array([0.9, 0, 0, 0], np.float32)
    far = ref + np.array([1.6, 0, 0, 0], np.float32)
    assert pr2_resize_ops.box_match(ref, near) == (2, 2, True)
    assert pr2_resize_ops.box_match(ref, far[:1]) == (0, 2, False)


def test_lap_variance_from_sums() -> None:
    # sums of a constant Laplacian of value 2 over 4 pixels: variance 0
    assert pr2_resize_ops.lap_variance([[8, 16]], [], (2, 2)) == [0.0]


def test_set_p_sample_is_seeded_hashed_and_verified(tmp_path: Path) -> None:
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    (src / 'sub').mkdir(parents=True)
    for i in range(6):
        _photo(src / 'sub' / f'{i}.jpg', seed=i)
    rel = set_p_sample.list_jpegs(src)
    assert set_p_sample.choose(rel, 3, 7) == set_p_sample.choose(rel, 3, 7)
    assert set_p_sample.choose(rel, 3, 7) != set_p_sample.choose(rel, 3, 8)
    rows = set_p_sample.copy_sample(src, dst, set_p_sample.choose(rel, 3, 7))
    assert len(rows) == 3
    for row in rows:
        name = str(row['name'])
        assert name == f'{str(row["sha256"])[:16]}.jpg'
        assert set_p_sample.sha256_file(dst / name) == row['sha256']


def test_exif_fixtures_cover_every_registered_kind(tmp_path: Path) -> None:
    img = Image.new('RGB', (64, 48), (200, 30, 30))
    fixtures = exif_fixtures.make_fixtures(img)
    assert {f'orient{t}' for t in range(1, 9)} <= set(fixtures)
    assert {
        'progressive',
        'truncated',
        'cmyk',
        'grey',
        'chroma444',
        'chroma422',
        'chroma420',
        'restart',
    } <= set(fixtures)
    with Image.open(io.BytesIO(fixtures['orient6'])) as o:
        assert o.getexif()[0x0112] == 6
    with Image.open(io.BytesIO(fixtures['progressive'])) as p:
        assert p.info.get('progressive') or p.info.get('progression')
    assert len(fixtures['truncated']) < len(fixtures['baseline'])


def test_kernel_busy_takes_the_union_of_overlapping_kernels(tmp_path: Path) -> None:
    db = tmp_path / 'trace.sqlite'
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER)')
    con.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER)')
    # two overlapping kernels (0-4 ms, 2-6 ms) and one later (8-10 ms): union 8 ms of a 10 ms window
    con.executemany(
        'INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?)',
        [(0, 4_000_000), (2_000_000, 6_000_000), (8_000_000, 10_000_000)],
    )
    con.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (1000000, 2000000)')
    con.commit()
    con.close()
    out = nsys_stats.kernel_busy(db)
    assert out['kernel_ms'] == pytest.approx(10.0)
    assert out['memcpy_ms'] == pytest.approx(1.0)
    assert out['window_ms'] == pytest.approx(10.0)
    assert out['busy_share'] == pytest.approx(0.8)
