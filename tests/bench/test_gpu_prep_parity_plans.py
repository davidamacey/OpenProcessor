"""``scripts/bench/gpu_prep_parity.py``, the N0 extension: the plans it sends, the rows of plan 17
section 1.2 with the engine-against-itself floor, the crop counts the load generator replays and
the comparison of what two engines handed the capture models."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.bench import frame_engine_stand_ins as si, gpu_prep_parity as par
from src.domain import frame_engine_wire as wire


REPO = Path(__file__).resolve().parents[2]
PLAN_CASES = json.loads((REPO / 'contracts/json/frame_engine_plan_cases.json').read_text())['cases']
REAL = ('yolov11_small_trt_end2end', 'pe_image_encoder', 'pe_crop_encoder')


def plan(kind: str, models: str) -> dict[str, Any]:
    text = par.build_plan(kind, models, 'yolov11_small_trt_end2end', 640, 114)  # type: ignore[arg-type]
    wire.parse_plan(text, has_boxes=False)  # the engine's own rule accepts it
    parsed: dict[str, Any] = json.loads(text)
    return parsed


def test_the_ingest_plan_is_the_apis_builder_not_a_second_copy() -> None:
    built = plan('ingest', 'real')
    # the contract's case of the plan "as built here", parsed (defaults filled in), with the
    # case's model names in place
    case = next(c for c in PLAN_CASES if c['id'] == 'ingest_plan_built_here')
    expected = json.loads(json.dumps(case['parsed']))
    expected['detector']['model'] = REAL[0]
    expected['embed']['model'] = REAL[1]
    expected['crop_embed']['model'] = REAL[2]
    assert built == expected


def test_the_ingest_plan_names_the_real_models_and_the_crop_recipe() -> None:
    p = plan('ingest', 'real')
    assert p['branches'] == ['detect', 'embed_frame', 'embed_crops']
    assert p['crops'] == 'inline'
    assert p['detector'] == {'model': REAL[0], 'input_size': 640, 'letterbox_fill': 114}
    assert p['embed']['model'] == REAL[1]
    assert p['embed']['size'] == 336
    assert p['crop_embed'] == {'model': REAL[2], 'size': 224, 'batch': 32} | {
        k: p['crop_embed'][k] for k in p['crop_embed'] if k not in ('model', 'size', 'batch')
    }


def test_the_frame_plan_is_the_detector_and_the_frame_vector_only() -> None:
    p = plan('frame', 'real')
    assert p['branches'] == ['detect', 'embed_frame']
    assert 'crop_embed' not in p
    assert p['crops'] == 'none'
    assert p['detector'] == plan('ingest', 'real')['detector']


@pytest.mark.parametrize('models', ['null', 'capture'])
def test_stand_in_plans_change_only_the_model_names(models: str) -> None:
    names = si.model_names(models)  # type: ignore[arg-type]
    for kind in ('frame', 'ingest'):
        real, stand_in = plan(kind, 'real'), plan(kind, models)
        assert stand_in['detector']['model'] == names['detector']
        assert stand_in['embed']['model'] == names['frame']
        if kind == 'ingest':
            assert stand_in['crop_embed']['model'] == names['crop']
            stand_in['crop_embed']['model'] = real['crop_embed']['model']
        stand_in['detector']['model'] = real['detector']['model']
        stand_in['embed']['model'] = real['embed']['model']
        assert stand_in == real


def test_the_plan_subcommand_writes_the_text_the_capture_sends(tmp_path: Path) -> None:
    out = tmp_path / 'plan.json'
    assert par.main(['plan', '--plan', 'ingest', '--models', 'null', '--out', str(out)]) == 0
    assert out.read_text() == par.build_plan(
        'ingest', 'null', 'yolov11_small_trt_end2end', 640, 114
    )


def test_the_plan_subcommand_creates_the_directory_of_its_output(tmp_path: Path) -> None:
    # The hold script asks for `<out>.d/plans/<name>.json` before anything else wrote there.
    out = tmp_path / 'hold.jsonl.d' / 'plans' / 'frame_real.json'
    assert par.main(['plan', '--plan', 'frame', '--models', 'real', '--out', str(out)]) == 0
    assert out.is_file()


# ------------------------------------------------------------------ the rows of section 1.2


def jpeg(optimize: bool = False, colour: int = 120) -> bytes:
    """A noisy 64 x 64 photo; the same arguments always give the same bytes."""
    noise = np.random.default_rng(colour).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    out = io.BytesIO()
    Image.fromarray(noise).save(out, 'JPEG', quality=90, optimize=optimize)
    return out.getvalue()


def row(**changes: Any) -> dict[str, np.ndarray]:
    base: dict[str, np.ndarray] = {
        'status': np.array([b'ok'], dtype=object),
        'decode_route': np.array([b'gpu'], dtype=object),
        'branch_status': np.array([b'{"detect": {"status": "ok"}}'], dtype=object),
        'size': np.array([640, 480], np.int32),
        'letterbox': np.array([1.0, 0.0, 80.0]),
        'det_boxes': np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.9, 0.9]], np.float32),
        'det_scores': np.array([0.5, 0.9], np.float32),
        'det_classes': np.array([1, 2], np.int32),
        'whole_embedding': np.array([1.0, 2.0, 3.0], np.float32),
        'crop_embeddings': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32),
        'crop_embedded': np.array([True, True]),
        'blur_sums': np.array([[100, 1, 2], [10, 1, 2], [20, 3, 4]], np.int64),
        'crop_jpegs': np.array([jpeg(), jpeg()], dtype=object),
    }
    base.update(changes)
    return base


def test_the_decode_route_the_branch_status_and_the_embedded_flags_are_rows_too() -> None:
    same = par.compare_pair(row(), row())
    assert (same['route_equal'], same['branch_equal'], same['embedded_equal']) == (True, True, True)
    other_route = row(decode_route=np.array([b'cpu_small'], dtype=object))
    assert par.compare_pair(row(), other_route)['route_equal'] is False
    assert par.compare_pair(row(), other_route)['bytes_equal'] is False
    failed = row(branch_status=np.array([b'{"detect": {"status": "error"}}'], dtype=object))
    assert par.compare_pair(row(), failed)['branch_equal'] is False
    fewer = row(crop_embedded=np.array([True, False]))
    assert par.compare_pair(row(), fewer)['embedded_equal'] is False


def test_the_frame_plan_compares_without_crop_tensors() -> None:
    def frame() -> dict[str, np.ndarray]:
        r = row()
        for k in ('crop_embeddings', 'crop_embedded', 'crop_jpegs'):
            del r[k]
        return r

    pair = par.compare_pair(frame(), frame())
    assert pair['bytes_equal'] is True
    assert 'crop_cos_min' not in pair
    s = par.summarize([pair])
    assert (s['crop_vectors'], s['jpeg_differs']) == (0, 0)


def test_vectors_are_counted_bit_identical_one_by_one_with_their_cosine() -> None:
    near = row(crop_embeddings=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0001]], np.float32))
    pair = par.compare_pair(row(), near)
    assert (pair['crop_vectors'], pair['crop_vectors_equal']) == (2, 1)
    assert pair['whole_equal'] is True
    assert float(pair['crop_cos_min']) > 0.999999
    s = par.summarize([pair, par.compare_pair(row(), row())])
    assert (s['crop_vectors'], s['crop_vectors_bit_identical']) == (4, 3)


def test_crop_jpegs_with_other_bytes_and_the_same_pixels_are_told_apart() -> None:
    # the same pixels coded with optimised tables: other bytes, other size, identical decode
    b = row(crop_jpegs=np.array([jpeg(optimize=True), jpeg(optimize=True)], dtype=object))
    pair = par.compare_pair(row(), b)
    assert pair['crop_jpeg_equal'] is False
    assert pair['crop_jpeg_pixels_equal'] is True
    assert pair['crop_jpeg_count_equal'] is True
    assert 0.0 < float(pair['crop_jpeg_size_ratio']) < 0.5
    other = row(crop_jpegs=np.array([jpeg(colour=240), jpeg()], dtype=object))
    assert par.compare_pair(row(), other)['crop_jpeg_pixels_equal'] is False
    fewer = row(crop_jpegs=np.array([jpeg()], dtype=object))
    assert par.compare_pair(row(), fewer)['crop_jpeg_count_equal'] is False


def summary_of(*pairs: dict[str, Any]) -> dict[str, float]:
    return par.summarize(list(pairs))


def passed(rows: list[dict[str, Any]]) -> dict[str, bool]:
    return {r['row']: r['pass'] for r in rows}


def test_a_matching_native_engine_passes_every_row() -> None:
    s = summary_of(par.compare_pair(row(), row()), par.compare_pair(row(), row()))
    rows = par.verdict(s, s, in_flight=1)
    assert all(r['pass'] for r in rows), rows


def test_a_vector_row_passes_on_the_floor_and_fails_below_it() -> None:
    noisy = row(crop_embeddings=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0001]], np.float32))
    floor = summary_of(par.compare_pair(row(), noisy), par.compare_pair(row(), row()))
    equal = summary_of(par.compare_pair(row(), row()), par.compare_pair(row(), row()))
    # the engine against the floor engine: no worse
    assert passed(par.verdict(equal, floor, 32))['crop_vectors_bit_identical_share']
    # a native engine whose vectors differ more often than the floor's two runs do
    worse = row(crop_embeddings=np.array([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1]], np.float32))
    bad = summary_of(par.compare_pair(row(), worse), par.compare_pair(row(), worse))
    got = passed(par.verdict(bad, floor, 32))
    assert not got['crop_vectors_bit_identical_share']
    assert not got['crop_cos_min']


def test_detections_are_exact_at_one_in_flight_and_floored_at_32() -> None:
    moved = row(det_boxes=np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.9, 0.91]], np.float32))
    s = summary_of(par.compare_pair(row(), row()), par.compare_pair(row(), moved))
    clean = summary_of(par.compare_pair(row(), row()), par.compare_pair(row(), row()))
    assert not passed(par.verdict(s, clean, 1))['detections_identical_share']
    # at 32 in flight a floor that is itself half-identical lets the same share pass
    assert passed(par.verdict(s, s, 32))['detections_identical_share']
    assert not passed(par.verdict(s, clean, 32))['detections_identical_share']


def test_exact_rows_fail_on_any_difference() -> None:
    s = summary_of(
        par.compare_pair(row(), row(decode_route=np.array([b'cpu_small'], dtype=object)))
    )
    clean = summary_of(par.compare_pair(row(), row()))
    assert not passed(par.verdict(s, clean, 32))['route_differs']
    blur = row(blur_sums=np.array([[101, 1, 2], [10, 1, 2], [20, 3, 4]], np.int64))
    assert not passed(par.verdict(summary_of(par.compare_pair(row(), blur)), clean, 32))[
        'blur_differs'
    ]


def test_crop_jpeg_bytes_may_differ_only_within_half_a_percent() -> None:
    clean = summary_of(par.compare_pair(row(), row()))
    optimised = row(crop_jpegs=np.array([jpeg(optimize=True)] * 2, dtype=object))
    s = summary_of(par.compare_pair(row(), optimised))
    assert s['jpeg_size_ratio_max'] > par.JPEG_SIZE_TOLERANCE
    assert not passed(par.verdict(s, clean, 1))['crop_jpeg_bytes_or_size']
    assert passed(par.verdict(clean, clean, 1))['crop_jpeg_bytes_or_size']
    assert passed(par.verdict(s, clean, 1))['jpeg_pixels_differ']


# ------------------------------------------------------------------------ crop counts


def test_crop_counts_carry_the_lists_sha256(tmp_path: Path) -> None:
    listing = tmp_path / 'list.txt'
    listing.write_text('/a.jpg\n/b.jpg\n/c.jpg\n')
    rows = [
        row(),
        row(crop_embedded=np.array([True, False, True])),
        row(crop_embedded=np.array([], bool)),
    ]
    out = tmp_path / 'counts.json'
    par.write_crop_counts(out, listing, rows)
    payload = json.loads(out.read_text())
    assert payload == {
        'list_sha256': hashlib.sha256(listing.read_bytes()).hexdigest(),
        'crops': [2, 2, 0],
    }
    frame_row = {k: v for k, v in row().items() if k != 'crop_embedded'}
    assert par.crop_counts([frame_row]) == [0]


def test_counts_cannot_be_recorded_for_a_limited_list(tmp_path: Path) -> None:
    listing = tmp_path / 'list.txt'
    listing.write_text('/a.jpg\n')
    with pytest.raises(SystemExit):
        par.main(['capture', '--url', 'x:1', '--images', str(listing), '--limit', '1',
                  '--out', str(tmp_path / 'o.npz'), '--crop-counts-out', str(tmp_path / 'c.json')])  # fmt: skip


# --------------------------------------------------------------------- compare-inputs


def capture_dir(root: Path, rows: list[bytes], model: str = 'capture_detector_640') -> Path:
    """What the capture model writes: one file per distinct row and the arrival-order index."""
    (root / model).mkdir(parents=True, exist_ok=True)
    lines = []
    for order, data in enumerate(rows):
        digest = hashlib.sha256(data).hexdigest()
        (root / model / f'{digest}.bin').write_bytes(data)
        lines.append(f'{order}\t{digest}\t{len(data)}\n')
    (root / model / 'index.txt').write_text(''.join(lines))
    return root


def test_the_same_rows_in_another_order_are_equal(tmp_path: Path) -> None:
    rows = [bytes([k]) * 8 for k in range(5)]
    a = capture_dir(tmp_path / 'a', rows)
    b = capture_dir(tmp_path / 'b', rows[::-1])
    report = par.compare_inputs(a, b)
    assert report['equal'] is True
    assert report['models']['capture_detector_640']['rows_a'] == 5


def test_a_differing_row_is_matched_by_order_and_diffed_byte_by_byte(tmp_path: Path) -> None:
    rows = [bytes([k]) * 8 for k in range(4)]
    other = list(rows)
    other[2] = bytes([2, 2, 9, 2, 2, 2, 2, 2])
    report = par.compare_inputs(
        capture_dir(tmp_path / 'a', rows), capture_dir(tmp_path / 'b', other)
    )
    model = report['models']['capture_detector_640']
    assert report['equal'] is False
    assert (model['only_a'], model['only_b']) == (1, 1)
    assert model['diffs'] == [
        {'order_a': 2, 'order_b': 2, 'differing_bytes': 1, 'max_abs': 7, 'first_offset': 2}
    ]


def test_a_repeated_row_counts_by_multiplicity(tmp_path: Path) -> None:
    row_a = bytes([1]) * 4
    a = capture_dir(tmp_path / 'a', [row_a, row_a, row_a])
    b = capture_dir(tmp_path / 'b', [row_a, row_a])
    model = par.compare_inputs(a, b)['models']['capture_detector_640']
    assert model['equal'] is False
    assert (model['rows_a'], model['rows_b'], model['only_a'], model['only_b']) == (3, 2, 1, 0)


def test_a_model_captured_in_one_directory_only_is_unequal(tmp_path: Path) -> None:
    a = capture_dir(tmp_path / 'a', [b'x' * 4])
    capture_dir(a, [b'y' * 4], model='capture_encoder_224')
    b = capture_dir(tmp_path / 'b', [b'x' * 4])
    report = par.compare_inputs(a, b)
    assert report['equal'] is False
    assert report['models']['capture_encoder_224']['error'] == 'captured in one directory only'
    assert report['models']['capture_detector_640']['equal'] is True
    assert par.compare_inputs(tmp_path / 'none_a', tmp_path / 'none_b')['equal'] is False


def test_the_compare_inputs_command_exits_nonzero_on_a_difference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = capture_dir(tmp_path / 'a', [b'x' * 4])
    b = capture_dir(tmp_path / 'b', [b'z' * 4])
    assert par.main(['compare-inputs', str(a), str(a)]) == 0
    assert par.main(['compare-inputs', str(a), str(b)]) == 1
    assert json.loads(capsys.readouterr().out.split('\n}\n{', 1)[0] + '\n}')['equal'] is True


def test_the_source_document_is_the_text_the_rust_client_writes_for_every_built_case() -> None:
    cases = json.loads((REPO / 'contracts/json/frame_engine_source_cases.json').read_text())[
        'cases'
    ]
    built = [c for c in cases if 'built' in c and c['key'].startswith('source/')]
    assert len(built) >= 4
    for case in built:
        # a file of the size and hash the case claims is not at hand: the document of a real one
        data = bytes(range(256)) * 3
        root = Path('/data/photos')
        text = par.source_document(root / case['key'].removeprefix('source/'), root, data)
        doc = json.loads(text)
        assert doc['key'] == case['key']
        assert (doc['size'], len(doc['imohash'])) == (len(data), 32)
        assert text == json.dumps(doc, sort_keys=True)
        assert list(doc) == ['imohash', 'key', 'size'], (
            'sorted keys, as the Rust builder writes them'
        )
