"""Compare two parity captures: metrics, gate evaluation, exit code and the markdown summary.

``compute_metrics`` pairs the captures item by item (image, box overlap and class name, never by
stored id) and measures each stage in the downstream order: detection, crops, embeddings,
core-route models, clustering, labels. ``evaluate`` applies the thresholds of one profile of
``parity_gates.json``. A gate whose metric could not be measured (a section missing from either
capture) is reported as SKIP and fails only under ``--strict``.
"""

from __future__ import annotations

import json
import operator
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.bench import parity_metrics as pm
from scripts.bench.parity_store import Capture, CaptureError


if TYPE_CHECKING:
    from pathlib import Path

OPS = {'>=': operator.ge, '<=': operator.le, '==': operator.eq}
GATE_SCHEMA = 1


def load_gates(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding='utf-8'))
    if doc.get('schema') != GATE_SCHEMA:
        raise ValueError(f'{path}: gate schema {doc.get("schema")}, this tool reads {GATE_SCHEMA}')
    for name, profile in doc['profiles'].items():
        for check in profile['checks']:
            if check['op'] not in OPS:
                raise ValueError(f'{path}: profile {name} check {check["id"]}: bad operator')
    return doc


def _pair_by_name(ref: np.ndarray, cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays of the names present in both (one row per name, first occurrence)."""
    cand_index: dict[str, int] = {}
    for i, name in enumerate(cand.tolist()):
        cand_index.setdefault(name, i)
    seen: set[str] = set()
    ri, ci = [], []
    for i, name in enumerate(ref.tolist()):
        if name in cand_index and name not in seen:
            seen.add(name)
            ri.append(i)
            ci.append(cand_index[name])
    return np.array(ri, dtype=np.int64), np.array(ci, dtype=np.int64)


def _vector_section(ref: np.ndarray, cand: np.ndarray, expected: int) -> dict[str, Any] | None:
    """Cosine summary of row-aligned vectors; ``expected`` rows existed in the reference.

    No expected rows means nothing to compare (``None``). Expected rows with no counterpart
    score the worst case, so a vanished section fails its gates instead of skipping them.
    """
    if expected == 0:
        return None
    if len(ref) == 0:
        return {
            'n': 0,
            'min': -1.0,
            'mean': -1.0,
            'p01': -1.0,
            'median': -1.0,
            'identical_share': 0.0,
        }
    if ref.shape[1] != cand.shape[1]:
        raise CaptureError(f'vector widths differ: {ref.shape[1]} against {cand.shape[1]}')
    return {
        **pm.cosine_summary(pm.cosine_rows(ref, cand)),
        'identical_share': float(np.mean(np.all(ref == cand, axis=1))),
    }


def _cluster_section(
    ref: Capture, cand: Capture, pairs: np.ndarray, has_vec: np.ndarray
) -> dict[str, Any] | None:
    ri, ci = pairs[has_vec, 0], pairs[has_vec, 1]
    a, b = ref.arrays['item_cluster'][ri], cand.arrays['item_cluster'][ci]
    ok = (a >= 0) & (b >= 0)
    if ok.sum() < 2:
        return None
    ri, ci, a, b = ri[ok], ci[ok], a[ok], b[ok]
    rv, cv = ref.arrays['item_vec'][ri], cand.arrays['item_vec'][ci]
    obj_ref, obj_cand = pm.kmeans_objective(rv, a), pm.kmeans_objective(cv, b)
    out: dict[str, Any] = {
        'items': int(ok.sum()),
        'clusters_ref': len(np.unique(a)),
        'clusters_cand': len(np.unique(b)),
        'ari': pm.adjusted_rand_index(a, b),
        'nmi': pm.normalized_mutual_info(a, b),
        'assignment_agreement': pm.assignment_agreement(rv, cv, a),
        'objective_ref': obj_ref,
        'objective_cand': obj_cand,
        'objective_rel_diff': (obj_cand - obj_ref) / obj_ref if obj_ref else 0.0,
    }
    sim = np.abs(
        ref.arrays['item_cluster_similarity'][ri] - cand.arrays['item_cluster_similarity'][ci]
    )
    if not np.isnan(sim).all():
        out['similarity_abs_delta_mean'] = float(np.nanmean(sim))
        out['similarity_abs_delta_p99'] = float(np.nanpercentile(sim, 99))
    out['core_agreement'] = float(
        np.mean(ref.arrays['item_cluster_core'][ri] == cand.arrays['item_cluster_core'][ci])
    )
    gt_ref, gt_cand = ref.arrays['item_gt'][ri], cand.arrays['item_gt'][ci]
    if (gt_ref != '').any() and (gt_cand != '').any():
        out['purity_ref'] = pm.purity(a, gt_ref)
        out['purity_cand'] = pm.purity(b, gt_cand)
        out['purity_drop_points'] = (out['purity_ref'] - out['purity_cand']) * 100.0
    return out


def _label_section(
    ref: Capture, cand: Capture, pairs: np.ndarray, *, with_vlm: bool
) -> dict[str, Any]:
    ri, ci = pairs[:, 0], pairs[:, 1]
    out: dict[str, Any] = {
        'class_agreement': pm.agreement_share(
            ref.arrays['item_label'][ri].tolist(), cand.arrays['item_label'][ci].tolist()
        ),
        'pairs': len(pairs),
    }
    if not with_vlm:
        return out
    rv, cv = ref.arrays['item_vlm'][ri], cand.arrays['item_vlm'][ci]
    both = (rv != '') | (cv != '')
    if both.any():
        out['vlm_agreement'] = pm.agreement_share(rv[both].tolist(), cv[both].tolist())
        out['vlm_pairs'] = int(both.sum())
    return out


def compute_metrics(ref: Capture, cand: Capture, settings: dict[str, Any]) -> dict[str, Any]:
    """Every measurable stage metric of ``cand`` against ``ref`` as nested plain dicts.

    A stage not captured in both (``sections`` in the metadata) is left out, so its gates SKIP.
    """
    for key in ('sha256', 'count'):
        if ref.meta['manifest'][key] != cand.meta['manifest'][key]:
            raise CaptureError(f'the captures differ in dataset manifest {key}')
    ra, ca = ref.arrays, cand.arrays
    both = set(ref.meta['sections']) & set(cand.meta['sections'])
    metrics: dict[str, Any] = {}
    pairs = pm.pair_items(
        (ra['item_image'], ra['item_box'], ra['item_class']),
        (ca['item_image'], ca['item_box'], ca['item_class']),
    )
    boxes = pm.box_metrics(
        (ra['item_image'], ra['item_box'], ra['item_class'], ra['item_score']),
        (ca['item_image'], ca['item_box'], ca['item_class'], ca['item_score']),
    )
    same_rect = np.all(
        ra['item_crop_rect'][pairs[:, 0]] == ca['item_crop_rect'][pairs[:, 1]], axis=1
    )
    boxes['crop_rect_identical_share'] = float(same_rect.mean()) if len(pairs) else 1.0
    metrics['boxes'] = boxes
    if len(pairs) == 0:
        return metrics
    has_vec = ra['item_has_vec'][pairs[:, 0]] & ca['item_has_vec'][pairs[:, 1]]
    ri, ci = pairs[has_vec, 0], pairs[has_vec, 1]
    expected = int(ra['item_has_vec'].sum())
    if section := _vector_section(ra['item_vec'][ri], ca['item_vec'][ci], expected):
        metrics['pe_crop'] = {**section, 'matched': len(ri)}
        if len(ri) > 2:
            metrics['knn'] = pm.knn_agreement(
                ra['item_vec'][ri],
                ca['item_vec'][ci],
                k=settings['knn_k'],
                n_queries=settings['knn_queries'],
                seed=settings['seed'],
                dup_threshold=settings['near_duplicate_cosine'],
                labels=ra['item_class'][ri],
                truth=(ra['item_gt'][ri], ca['item_gt'][ci]),
            )
    if 'frames' in both:
        fr, fc = _pair_by_name(ra['frame_image'], ca['frame_image'])
        if section := _vector_section(
            ra['frame_vec'][fr], ca['frame_vec'][fc], len(ra['frame_image'])
        ):
            metrics['pe_frame'] = {**section, 'matched': len(fr)}
    if 'clip' in both:
        cr, cc = _pair_by_name(ra['clip_image'], ca['clip_image'])
        if section := _vector_section(
            ra['clip_vec'][cr], ca['clip_vec'][cc], len(ra['clip_image'])
        ):
            metrics['clip'] = section
    if 'faces' in both:
        face_pairs = pm.pair_items(
            (ra['face_image'], ra['face_box'], np.full(len(ra['face_image']), 'face')),
            (ca['face_image'], ca['face_box'], np.full(len(ca['face_image']), 'face')),
        )
        if section := _vector_section(
            ra['face_vec'][face_pairs[:, 0]],
            ca['face_vec'][face_pairs[:, 1]],
            len(ra['face_image']),
        ):
            metrics['face'] = {
                **section,
                'ref': len(ra['face_image']),
                'cand': len(ca['face_image']),
            }
    if 'ocr' in both and (len(ra['ocr_image']) or len(ca['ocr_image'])):
        metrics['ocr'] = pm.text_agreement(
            (ra['ocr_image'], ra['ocr_box'], ra['ocr_text']),
            (ca['ocr_image'], ca['ocr_box'], ca['ocr_text']),
        )
    if 'clusters' in both and (cluster := _cluster_section(ref, cand, pairs, has_vec)):
        metrics['cluster'] = cluster
    metrics['labels'] = _label_section(ref, cand, pairs, with_vlm='vlm' in both)
    return metrics


def _lookup(metrics: dict[str, Any], dotted: str) -> Any:
    node: Any = metrics
    for part in dotted.split('.'):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def evaluate(metrics: dict[str, Any], gates: dict[str, Any], profile: str) -> dict[str, Any]:
    """Apply one profile of the gate file to the metrics."""
    spec = gates['profiles'][profile]
    checks = []
    for check in spec['checks']:
        observed = _lookup(metrics, check['metric'])
        if observed is None:
            status = 'SKIP'
        else:
            status = 'PASS' if OPS[check['op']](observed, check['value']) else 'FAIL'
        checks.append({**check, 'observed': observed, 'status': status})
    count = lambda s: sum(1 for c in checks if c['status'] == s)  # noqa: E731
    return {
        'profile': profile,
        'description': spec['description'],
        'checks': checks,
        'recorded': {m: _lookup(metrics, m) for m in spec['record']},
        'failed': count('FAIL'),
        'skipped': count('SKIP'),
        'passed': count('FAIL') == 0,
    }


def exit_code(result: dict[str, Any], *, strict: bool) -> int:
    return 1 if result['failed'] or (strict and result['skipped']) else 0


def _fmt(value: Any) -> str:
    if value is None:
        return 'n/a'
    return f'{value:.5f}' if isinstance(value, float) else str(value)


def render_markdown(
    ref_meta: dict[str, Any],
    cand_meta: dict[str, Any],
    metrics: dict[str, Any],
    result: dict[str, Any],
) -> str:
    verdict = 'PASS' if result['passed'] else 'FAIL'
    lines = [
        f'# Parity: {cand_meta["tag"]} against {ref_meta["tag"]} ({verdict})',
        '',
        f'Profile `{result["profile"]}`: {result["description"]}',
        '',
        '| | reference | candidate |',
        '|---|---|---|',
        f'| tag | {ref_meta["tag"]} | {cand_meta["tag"]} |',
        f'| engine | {ref_meta["engine"].get("label", "")} | {cand_meta["engine"].get("label", "")} |',
        f'| git sha | {ref_meta["environment"].get("git_sha", "")[:12]} | '
        f'{cand_meta["environment"].get("git_sha", "")[:12]} |',
        f'| manifest sha256 | {ref_meta["manifest"]["sha256"][:16]} | {cand_meta["manifest"]["sha256"][:16]} |',
        '',
        '| Gate | Metric | Observed | Rule | Status |',
        '|---|---|---:|---|---|',
    ]
    lines += [
        f'| {c["id"]} | `{c["metric"]}` | {_fmt(c["observed"])} | {c["op"]} {c["value"]} | {c["status"]} |'
        for c in result['checks']
    ]
    if result['recorded']:
        lines += ['', 'Recorded, not gated:', '', '| Metric | Value |', '|---|---:|']
        lines += [f'| `{k}` | {_fmt(v)} |' for k, v in result['recorded'].items()]
    lines += [
        '',
        f'Items: {metrics["boxes"]["items_ref"]} reference, {metrics["boxes"]["items_cand"]} candidate, '
        f'{metrics["boxes"]["matched"]} matched.',
        '',
    ]
    return '\n'.join(lines)
