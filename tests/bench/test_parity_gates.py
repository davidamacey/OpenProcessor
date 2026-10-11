"""compare: metrics between two synthetic captures, gate evaluation, exit-code semantics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.bench import parity_gates as pg, parity_store as ps


GATES = Path(pg.__file__).with_name('parity_gates.json')


def _evaluate(ref, cand, profile: str = 'precision') -> dict:
    gates = pg.load_gates(GATES)
    metrics = pg.compute_metrics(ref, cand, gates['settings'])
    return pg.evaluate(metrics, gates, profile)


def _status(result: dict) -> dict[str, str]:
    return {c['id']: c['status'] for c in result['checks']}


def _noise(scale: float, seed: int = 9):
    def apply(a: dict[str, np.ndarray]) -> None:
        rng = np.random.default_rng(seed)
        v = a['item_vec'].astype(np.float32)
        v += rng.normal(scale=scale, size=v.shape).astype(np.float32)
        a['item_vec'] = v.astype(np.float16)
        f = a['frame_vec'].astype(np.float32)
        f += rng.normal(scale=scale, size=f.shape).astype(np.float32)
        a['frame_vec'] = f.astype(np.float16)

    return apply


def test_a_capture_against_itself_passes_every_gate(make_capture) -> None:
    result = _evaluate(make_capture('ref'), make_capture('cand'))
    assert result['failed'] == 0
    assert result['skipped'] == 0
    assert result['passed'] is True
    assert result['recorded']['pe_crop.identical_share'] == 1.0


def test_half_precision_noise_passes_and_is_visible_in_the_recorded_metrics(make_capture) -> None:
    ref = make_capture('ref')
    cand = make_capture('cand', mutate=_noise(0.0005))
    result = _evaluate(ref, cand)
    assert result['passed'] is True
    assert result['recorded']['pe_crop.identical_share'] < 0.5


def test_heavy_vector_noise_fails_the_cosine_and_knn_gates(make_capture) -> None:
    ref = make_capture('ref')
    cand = make_capture('cand', mutate=_noise(0.5))
    status = _status(_evaluate(ref, cand))
    assert status['P-pe-p01'] == 'FAIL'
    assert status['P-knn'] == 'FAIL'
    assert status['P-cluster-assign'] == 'FAIL'


def test_shifted_boxes_fail_the_box_gate_and_change_the_crop_rectangles(make_capture) -> None:
    def shift(a: dict[str, np.ndarray]) -> None:
        a['item_box'] = a['item_box'] + np.float32([0.04, 0, 0.04, 0])
        a['item_crop_rect'] = a['item_crop_rect'] + 4

    result = _evaluate(make_capture('ref'), make_capture('cand', mutate=shift))
    assert _status(result)['P-box'] == 'FAIL'
    assert _status(result)['P-item'] == 'PASS'
    assert result['recorded']['boxes.crop_rect_identical_share'] == 0.0


def test_a_class_flip_is_not_hidden_by_box_overlap(make_capture) -> None:
    def flip(a: dict[str, np.ndarray]) -> None:
        a['item_class'] = a['item_class'].copy()
        a['item_class'][:20] = 'zebra'

    status = _status(_evaluate(make_capture('ref'), make_capture('cand', mutate=flip)))
    assert status['P-item'] == 'FAIL'
    assert status['P-box-confident'] == 'FAIL'


def test_a_permuted_cluster_labelling_still_passes(make_capture) -> None:
    def permute(a: dict[str, np.ndarray]) -> None:
        a['item_cluster'] = ((a['item_cluster'] + 2) % 4).astype(np.int32)

    result = _evaluate(make_capture('ref'), make_capture('cand', mutate=permute))
    status = _status(result)
    assert status['P-cluster-assign'] == 'PASS'
    assert status['P-cluster-purity'] == 'PASS'
    assert result['recorded']['cluster.ari'] == pytest.approx(1.0)


def test_random_clusters_lose_purity(make_capture) -> None:
    def scramble(a: dict[str, np.ndarray]) -> None:
        a['item_cluster'] = (
            np.random.default_rng(3).integers(0, 4, size=len(a['item_id'])).astype(np.int32)
        )

    result = _evaluate(make_capture('ref'), make_capture('cand', mutate=scramble))
    assert _status(result)['P-cluster-purity'] == 'FAIL'
    assert result['recorded']['cluster.ari'] < 0.2


def test_label_and_vlm_disagreement_fail_their_gates(make_capture) -> None:
    def relabel(a: dict[str, np.ndarray]) -> None:
        a['item_label'] = a['item_label'].copy()
        a['item_label'][:9] = 'x'
        a['item_vlm'] = a['item_vlm'].copy()
        a['item_vlm'][:5] = 'y'

    status = _status(_evaluate(make_capture('ref'), make_capture('cand', mutate=relabel)))
    assert status['P-label'] == 'FAIL'
    assert status['P-vlm'] == 'FAIL'


def test_ocr_and_face_changes_are_gated(make_capture) -> None:
    def change(a: dict[str, np.ndarray]) -> None:
        a['ocr_text'] = np.array(['changed'] * len(a['ocr_text']))
        a['face_vec'] = (-a['face_vec'].astype(np.float32)).astype(np.float16)
        a['clip_vec'] = (-a['clip_vec'].astype(np.float32)).astype(np.float16)

    status = _status(_evaluate(make_capture('ref'), make_capture('cand', mutate=change)))
    assert status['P-ocr'] == 'FAIL'
    assert status['P-face'] == 'FAIL'
    assert status['P-clip'] == 'FAIL'


def test_missing_sections_are_skipped_not_failed(make_capture) -> None:
    ref = make_capture('ref')
    cand = make_capture('cand')
    cand.arrays.update({k: v[:0] for k, v in cand.arrays.items() if k.startswith(('clip', 'ocr'))})
    cand.meta['sections'] = [s for s in cand.meta['sections'] if s not in ('clip', 'ocr')]
    result = _evaluate(ref, cand)
    status = _status(result)
    assert status['P-clip'] == 'SKIP'
    assert status['P-ocr'] == 'SKIP'
    assert result['passed'] is True
    assert pg.exit_code(result, strict=True) == 1
    assert pg.exit_code(result, strict=False) == 0


def test_a_section_that_vanished_from_a_captured_stage_fails(make_capture) -> None:
    cand = make_capture('cand')
    cand.arrays.update({k: v[:0] for k, v in cand.arrays.items() if k.startswith('ocr')})
    status = _status(_evaluate(make_capture('ref'), cand))
    assert status['P-ocr'] == 'FAIL'


def test_vlm_label_present_in_only_one_capture_disagrees(make_capture) -> None:
    def drop(a: dict[str, np.ndarray]) -> None:
        a['item_vlm'] = np.array([''] * len(a['item_vlm']), dtype='U8')

    status = _status(_evaluate(make_capture('ref'), make_capture('cand', mutate=drop)))
    assert status['P-vlm'] == 'FAIL'


def test_exit_code_is_nonzero_on_failure(make_capture) -> None:
    result = _evaluate(make_capture('ref'), make_capture('cand', mutate=_noise(0.5)))
    assert pg.exit_code(result, strict=False) == 1


def test_different_manifests_cannot_be_compared(make_capture) -> None:
    with pytest.raises(ps.CaptureError, match='manifest'):
        pg.compute_metrics(make_capture('a'), make_capture('b', manifest='other'), {})


def test_unknown_operator_and_metric_shape_are_rejected(tmp_path: Path) -> None:
    doc = json.loads(GATES.read_text())
    doc['profiles']['precision']['checks'][0]['op'] = '~='
    bad = tmp_path / 'g.json'
    bad.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='operator'):
        pg.load_gates(bad)


def test_shipped_gate_ids_are_unique_and_follow_the_plan() -> None:
    gates = pg.load_gates(GATES)
    ids = [c['id'] for c in gates['profiles']['precision']['checks']]
    assert len(ids) == len(set(ids))
    by_id = {c['id']: c for c in gates['profiles']['precision']['checks']}
    assert by_id['P-pe-p01']['value'] == 0.995
    assert by_id['P-pe-median']['value'] == 0.999
    assert by_id['P-cluster-assign']['value'] == 0.98
    assert by_id['P-cluster-purity']['value'] == 0.5
    assert by_id['P-knn']['value'] == 0.95


def test_markdown_lists_every_check_and_the_verdict(make_capture) -> None:
    ref, cand = make_capture('ref', engine='fp32'), make_capture('cand', engine='fp16')
    gates = pg.load_gates(GATES)
    metrics = pg.compute_metrics(ref, cand, gates['settings'])
    result = pg.evaluate(metrics, gates, 'precision')
    text = pg.render_markdown(ref.meta, cand.meta, metrics, result)
    assert 'PASS' in text
    assert 'fp32' in text
    assert 'fp16' in text
    assert all(c['id'] in text for c in result['checks'])


def test_objective_difference_is_relative_to_the_reference(make_capture) -> None:
    ref = make_capture('ref')
    cand = make_capture('cand', mutate=_noise(0.05))
    metrics = pg.compute_metrics(ref, cand, pg.load_gates(GATES)['settings'])['cluster']
    expected = (metrics['objective_cand'] - metrics['objective_ref']) / metrics['objective_ref']
    assert metrics['objective_rel_diff'] == pytest.approx(expected)
    assert metrics['objective_rel_diff'] > 0.1


def test_cluster_geometry_deltas_are_reported(make_capture) -> None:
    def shift(a: dict[str, np.ndarray]) -> None:
        a['item_cluster_similarity'] = a['item_cluster_similarity'] - np.float32(0.1)
        a['item_cluster_core'] = np.zeros_like(a['item_cluster_core'])

    ref, cand = make_capture('ref'), make_capture('cand', mutate=shift)
    cluster = pg.compute_metrics(ref, cand, pg.load_gates(GATES)['settings'])['cluster']
    assert cluster['similarity_abs_delta_mean'] == pytest.approx(0.1, abs=1e-5)
    assert cluster['core_agreement'] == 0.0
