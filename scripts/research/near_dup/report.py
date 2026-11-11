"""Condense the raw nd:P0..P2 outputs (in ``artifacts_local/near_dup/``) into the small JSON that backs the note.

``report.py --raw <artifacts dir> --out <docs/benchmarks/near_duplicate_detection>`` writes ``p0.json``,
``p1_summary.json`` and ``p2_summary.json`` plus ``SHA256SUMS``; nothing in them names a file or an image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    data: dict = json.loads(path.read_text())
    return data


def p0(raw: Path) -> dict:
    out: dict = {'note': 'COCO val2017 first 200 images (sorted by file name)'}
    for space in ('pe_frame', 'mobileclip_frame'):
        out[space] = {
            'venv_fp16_vs_venv_fp32': _load(raw / f'p0/precision_{space}.json')['fp16_vs_fp32'],
            'venv_vs_product_tensorrt_plan': _load(raw / f'p0/parity_engine_{space}.json'),
        }
    return out


def p1(raw: Path) -> dict:
    out: dict = {}
    for space in ('pe_frame', 'mobileclip_frame'):
        r = _load(raw / f'p1/p1_{space}.json')
        entry: dict = {
            'registration_commit': r['registration_commit'],
            'code_commit': r['code_commit'],
            'n_items': r['n_items'],
            'n_families': r['n_families'],
            'guard': r['guard'],
            'chosen_cut_per_seed': {
                rule: [s['chosen'][rule] for s in r['splits']]
                for rule in ('single', 'single_guard', 'complete')
            },
            'stability_single_cut_per_seed': [s['stability'] for s in r['splits']],
            'product_partition_equals_research_single': all(
                all(s['product_cross_check'].values()) for s in r['splits']
            ),
            'test_half': [dict(s['test']) | {'seed': s['seed']} for s in r['splits']],
            'calibration_single': [
                {
                    c: {
                        'false_group_rate': v['false_group_rate'],
                        'pooled_recall': v['pooled_recall']['recall'],
                        'T-q60': v['per_transform']['T-q60']['recall'],
                        'T-half': v['per_transform']['T-half']['recall'],
                        'T-crop90': v['per_transform']['T-crop90']['recall'],
                        'largest_group': v['largest_group'],
                    }
                    for c, v in s['calibration']['single'].items()
                }
                for s in r['splits']
            ],
            'copydays_single': {
                c: {
                    'strong_recall': v['per_transform']['copydays_strong']['recall'],
                    'strong_ci': v['per_transform']['copydays_strong']['ci'],
                    'false_group_rate': v['false_group_rate'],
                    'groups': v['groups'],
                }
                for c, v in r.get('copydays', {}).get('single', {}).items()
            },
        }
        out[space] = entry
    return out


def p2(raw: Path) -> dict:
    out: dict = {'exact': _load(raw / 'p2/exact.json')}
    for name in ('cost_hires', 'cost_coco', 'exact_cost_rerun'):
        f = raw / f'p2/{name}.json'
        if f.exists():
            doc = _load(f)
            out[name] = {k: v for k, v in doc.items() if k != 'rows'}
    perceptual = _load(raw / 'p2/perceptual.json')
    out['perceptual'] = {
        'pdq_quality_median': perceptual['pdq_quality_median'],
        'splits': [
            {
                'seed': sp['seed'],
                'base_cut': sp['base_cut'],
                'embedding_only': _lite(sp['embedding_only']),
                'hashes': {
                    h: {
                        t: {
                            'alone': _lite(d['alone'][t]),
                            'plus_embedding': _lite(d['plus_embedding'][t]),
                        }
                        for t in d['alone']
                        if int(t) % 4 == 0 or int(t) <= 12
                    }
                    for h, d in sp['hashes'].items()
                },
            }
            for sp in perceptual['splits']
        ],
    }
    return out


def _lite(m: dict) -> dict:
    return {
        'pooled_recall': m['pooled_recall']['recall'],
        'false_groups': m['false_groups'],
        'groups': m['groups'],
        'false_group_rate': m['false_group_rate'],
        'largest_group': m['largest_group'],
        'T-q60': m['per_transform']['T-q60']['recall'],
        'C-social': m['per_transform']['C-social']['recall'],
        'CB-half': m['per_transform']['CB-half']['recall'],
    }


def write_sums(out_dir: Path) -> None:
    lines = [
        f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}'
        for p in sorted(out_dir.glob('*.json'))
    ]
    (out_dir / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, fn in (('p0', p0), ('p1_summary', p1), ('p2_summary', p2)):
        (args.out / f'{name}.json').write_text(json.dumps(fn(args.raw), indent=1) + '\n')
    write_sums(args.out)
    sys.stdout.write(f'wrote {args.out}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
