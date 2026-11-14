"""Markdown tables for the face-grouping evaluation."""

from __future__ import annotations

from typing import Any


def _f(v: Any, nd: int = 4) -> str:
    return '-' if v is None else f'{v:.{nd}f}' if isinstance(v, float) else str(v)


def _params(p: dict[str, Any]) -> str:
    return ', '.join(f'{k}={v}' for k, v in p.items())


def _score_row(label: str, r: dict[str, Any] | None) -> str:
    if r is None:
        return f'| {label} | none reaches the floor | | | | | | | | |'
    cols = [
        'bcubed_precision',
        'bcubed_recall',
        'bcubed_f',
        'pair_precision',
        'pair_recall',
        'task_recall',
        'clusters',
        'singleton_share',
        'wrong_merge_rate',
        'split_rate',
    ]
    return f'| {label} | ' + ' | '.join(_f(r[c]) for c in cols) + ' |'


HEAD = (
    '| method (params) | BCubed P | BCubed R | BCubed F | pair P | pair R | task recall | clusters | '
    'singleton share | wrong-merge rate | split rate |\n|---|---|---|---|---|---|---|---|---|---|---|'
)


def markdown(res: dict[str, Any]) -> str:
    out = [
        '# Face grouping evaluation',
        '',
        '## Datasets',
        '',
        '| set | images | identities | zero-face images | multi-face images | npz sha256 | pairs sha256 |',
        '|---|---|---|---|---|---|---|',
    ]
    for n, d in res['datasets'].items():
        out.append(
            f'| {n} | {d["images"]} | {d["identities"]} | {d["images_zero_faces"]} | '
            f'{d["images_multiple_faces"]} | `{d["npz_sha256"][:16]}` | `{d["pairs_sha256"][:16]}` |'
        )
    out += [
        '',
        f'Identity overlap across sets: {res["overlap"]}',
        '',
        '## A. Verification (10-fold)',
        '',
        '| set | pairs | missing a face | accuracy | std | TAR@FAR 1e-2 | TAR@FAR 1e-3 |',
        '|---|---|---|---|---|---|---|',
    ]
    for n, v in res['verification'].items():
        out.append(
            f'| {n} | {v["pairs"]} | {v["pairs_missing_a_face"]} | {_f(v["accuracy"])} | {_f(v["std"])} | '
            f'{_f(v["tar_at_far_1e-2"])} | {_f(v["tar_at_far_1e-3"])} |'
        )
    for bed, b in res['beds'].items():
        out += ['', f'## B/C. {bed} ({b["faces"]} faces)']
        for floor in ('0.95', '0.99'):
            out += [
                '',
                f'Best point per method at BCubed precision >= {floor} (highest BCubed recall):',
                '',
                HEAD,
            ]
            for m, r in b[f'best_at_{floor}'].items():
                out.append(_score_row(f'{m} ({_params(r["params"])})' if r else m, r))
        today = next(
            r for r in b['sweep'] if r['method'] == 'single' and r['params'] == {'sim': 0.7}
        )
        out += [
            '',
            'Today (single linkage, cosine 0.7):',
            '',
            HEAD,
            _score_row('single (sim=0.7)', today),
        ]
    out += [
        '',
        f'## D. Quality signals (best method: {res["best_method_bed2_p95"]})',
        '',
        '| signal | pair error reject area | base error | wrong-merge reject area | base rate |',
        '|---|---|---|---|---|',
    ]
    for k, v in res['quality'].items():
        out.append(
            f'| {k} | {_f(v["pair_error_reject_area"])} | {_f(v["pair_base_error"])} | '
            f'{_f(v.get("wrong_merge_reject_area"))} | {_f(v.get("wrong_merge_base_rate"))} |'
        )
    out += [
        '',
        f'Gated clustering with `{res["quality_best_signal"]}` (weak faces may join, never found or bridge):',
        '',
        '| reject fraction | BCubed P | BCubed R | task recall | singleton share | wrong-merge rate |',
        '|---|---|---|---|---|---|',
    ]
    for r in res['gated']:
        out.append(
            f'| {r["reject_fraction"]} | {_f(r["bcubed_precision"])} | {_f(r["bcubed_recall"])} | '
            f'{_f(r["task_recall"])} | {_f(r["singleton_share"])} | {_f(r["wrong_merge_rate"])} |'
        )
    e = res['incremental']
    out += [
        '',
        '## E. Incremental stability (10 shuffled batches)',
        '',
        f'- ARI incremental vs one-shot: {_f(e["ari_incremental_vs_one_shot"])}',
        f'- faces of batches 1-9 whose cluster changed at batch 10, incremental: {_f(e["moved_share_incremental_batch9_to_10"])}',
        f'- same, re-clustering from scratch each time: {_f(e["moved_share_recluster_batch9_to_10"])}',
        '',
        HEAD,
        _score_row('incremental', e['incremental_scores']),
        _score_row('one-shot', e['one_shot_scores']),
        '',
    ]
    return '\n'.join(out)
