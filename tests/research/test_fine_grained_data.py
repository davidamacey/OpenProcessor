"""Set C splits, preprocessing variants and the FG-0 label-check logic."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from scripts.research.fine_grained import fg0, preprocess, setc


def _items() -> list[setc.Item]:
    out = []
    for make, models in {'A': ['A1', 'A2'], 'B': ['B1'], 'C': ['C1', 'C2', 'C3']}.items():
        for model in models:
            for gen in ('g1', 'g2'):
                out.extend(setc.Item(f'{model}_{gen}_{n}.jpg', make, model, gen) for n in range(7))
    return out


def test_halves_are_even_per_generation_and_seeded() -> None:
    items = _items()
    a = setc.split_halves(items, seed=1)
    assert a == setc.split_halves(list(reversed(items)), seed=1)
    assert a != setc.split_halves(items, seed=2)
    for model in {i.model for i in items}:
        for gen in ('g1', 'g2'):
            sides = [a[i.path] for i in items if i.model == model and i.generation == gen]
            assert abs(sides.count('calibration') - sides.count('heldout')) <= 1


def test_new_models_hold_out_half_of_each_multi_model_make_only() -> None:
    held = setc.split_new_models(_items(), seed=3)
    assert 'B1' not in held  # a one-model make has nothing to hold out
    assert len({m for m in held if m.startswith('A')}) == 1
    assert len({m for m in held if m.startswith('C')}) == 1  # 3 models -> 3 // 2 held out
    assert held == setc.split_new_models(_items(), seed=3)


def test_centre_matches_the_repo_pe_preprocessing() -> None:
    from src.utils.pe_preprocess import whole_frame_rgb_from_image

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (120, 200, 3), dtype=np.uint8))
    assert np.array_equal(preprocess.centre(img, 336), whole_frame_rgb_from_image(img, 336))


def test_letterbox_keeps_everything_and_pads_symmetrically() -> None:
    arr = np.zeros((40, 100, 3), dtype=np.uint8)
    arr[:] = (255, 0, 0)
    out = preprocess.letterbox(Image.fromarray(arr), 100)
    assert out.shape == (100, 100, 3)
    assert (out[0] == preprocess.PAD_RGB).all()
    assert (out[-1] == preprocess.PAD_RGB).all()
    assert (out[50] == (255, 0, 0)).all()  # the whole width survives, nothing is cropped
    rows = np.nonzero((out != preprocess.PAD_RGB).any(axis=(1, 2)))[0]
    assert abs(rows[0] - (99 - rows[-1])) <= 1


def test_grow_box_adds_ten_percent_per_side_and_clips() -> None:
    assert preprocess.grow_box((100, 100, 200, 300), 1000, 1000) == (90, 80, 210, 320)
    assert preprocess.grow_box((5, 5, 50, 50), 52, 1000) == (0, 0, 52, 55)


def test_variants_differ_on_a_wide_box_and_unknown_variant_fails() -> None:
    frame = Image.fromarray(
        np.random.default_rng(1).integers(0, 255, (300, 600, 3), dtype=np.uint8)
    )
    box = (100, 50, 500, 250)
    c = preprocess.variant_input(frame, box, 'centre', 64)
    lb = preprocess.variant_input(frame, box, 'letterbox', 64)
    m10 = preprocess.variant_input(frame, box, 'margin10', 64)
    assert not np.array_equal(c, lb)
    assert not np.array_equal(c, m10)
    with pytest.raises(ValueError, match='unknown variant'):
        preprocess.variant_input(frame, box, 'zoom', 64)


def _png(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (40, 30), color).save(buf, 'PNG')
    return buf.getvalue()


def test_plan_sheets_plants_come_from_outside_the_sample_with_the_right_make() -> None:
    items = _items()
    sample = [i.path for i in items if i.generation == 'g1' and i.path.endswith('_0.jpg')]
    specs = fg0.plan_sheets(items, sample, seed=0)
    by = {i.path: i for i in items}
    assert len(specs) == 6
    for s in specs:
        assert set(s.sample_paths) <= set(sample)
        for path, kind in s.plant_paths:
            assert path not in sample
            if kind == 'same_make':
                assert by[path].make == s.make
                assert by[path].model != s.model
            else:
                assert by[path].make != s.make
    b1 = next(s for s in specs if s.model == 'B1')
    assert [k for _, k in b1.plant_paths] == ['other_make']  # B has a single model


def test_parse_verdict_accepts_fenced_json_and_rejects_bad_replies() -> None:
    good = '```json\n{"cells": [{"cell": 1, "model": "yes", "make": "yes"}, {"cell": 2, "model": "no", "make": "unsure"}]}\n```'
    assert fg0.parse_verdict(2, good)[2] == {'model': 'no', 'make': 'unsure'}
    for bad in (
        'no json',
        json.dumps({'cells': [{'cell': 1, 'model': 'yes', 'make': 'yes'}]}),  # cell 2 missing
        json.dumps({'cells': [{'cell': 3, 'model': 'yes', 'make': 'yes'}] * 2}),  # out of range
        json.dumps({'cells': [{'cell': 1, 'model': 'maybe', 'make': 'yes'}] * 2}),
        json.dumps({'cells': [{'cell': 1, 'model': 'yes', 'make': 'yes'}] * 2}),  # duplicate
    ):
        with pytest.raises(fg0.VerdictError):
            fg0.parse_verdict(2, bad)


def _rows(
    sample_yes: int, sample_no: int, sample_unsure: int, plants_rejected: int, plants: int
) -> list[dict[str, str]]:
    rows = [{'role': 'sample', 'model': 'yes', 'make': 'yes'}] * sample_yes
    rows += [{'role': 'sample', 'model': 'no', 'make': 'yes'}] * sample_no
    rows += [{'role': 'sample', 'model': 'unsure', 'make': 'unsure'}] * sample_unsure
    rows += [{'role': 'same_make', 'model': 'no', 'make': 'yes'}] * plants_rejected
    rows += [{'role': 'same_make', 'model': 'yes', 'make': 'yes'}] * (plants - plants_rejected)
    return rows


def test_score_uses_decided_rows_and_needs_the_reviewer_to_catch_plants() -> None:
    ok = fg0.score(_rows(96, 4, 50, 9, 10))
    assert ok['model_accuracy']['decided']['rate'] == pytest.approx(0.96)
    assert ok['model_accuracy']['strict']['rate'] == pytest.approx(96 / 150)
    assert ok['fg0_verdict'] == 'PASS'
    assert fg0.score(_rows(90, 10, 0, 9, 10))['fg0_verdict'] == 'FAIL'
    # accuracy looks fine but the reviewer accepts most planted wrong models: it cannot read this level
    assert fg0.score(_rows(100, 0, 0, 3, 10))['fg0_verdict'] == 'INCONCLUSIVE'
