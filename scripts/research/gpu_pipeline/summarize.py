#!/usr/bin/env python3
"""Render the result tables of gpu:PR-0..PR-2 from the raw JSON files as Markdown.

summarize.py pr0 DIR | pr1 DIR | pr2 DIR      (DIR holds the files written by the run scripts)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median
from typing import Any


STAGES = (
    'read',
    'decode',
    'letterbox',
    'detect',
    'embed_frame',
    'embed_crops',
    'embed_pe',
    'blur',
    'crop_copy',
    'crop_jpeg',
)


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _row(cells: list[Any]) -> str:
    return '| ' + ' | '.join(str(c) for c in cells) + ' |'


def pr0(directory: Path) -> str:
    out = [
        _row(
            [
                'set',
                'images/s per rep',
                'spread',
                'declined',
                'load at start',
                'NVML util %',
                'kernel-busy share',
                'PE share of GPU ms',
                'GPU kernel ms/image',
            ]
        ),
        _row(['---'] * 9),
    ]
    stage_rows = [_row(['set', *STAGES]), _row(['---'] * (len(STAGES) + 1))]
    for label in ('A', 'B', 'P'):
        run, prof = directory / f'pr0_run_{label}.json', directory / f'pr0_profile_{label}.json'
        if not run.exists() or not prof.exists():
            continue
        r, p = _load(run), _load(prof)
        ips = [x['images_per_s'] for x in r['reps']]
        spread = (max(ips) - min(ips)) / median(ips)
        out.append(
            _row(
                [
                    label,
                    ' / '.join(f'{x:.1f}' for x in ips),
                    f'{spread:.2f}',
                    r['reps'][0]['declined'],
                    ' / '.join(f'{x["load_1m"]:.0f}' for x in r['reps']),
                    ' / '.join(f'{x["nvml"]["gpu_util_mean"]:.0f}' for x in r['reps']),
                    f'{p["busy"]["busy_share"]:.2f}',
                    f'{p["pe_share_of_gpu_ms"]:.2f}',
                    f'{p["gpu_kernel_ms_per_image_total"]:.1f}',
                ]
            )
        )
        g = p['gpu_kernel_ms_per_image_by_stage']
        stage_rows.append(_row([label, *[f'{g[s]:.2f}' for s in STAGES]]))
    return '\n'.join(
        [*out, '', 'GPU kernel ms per image by stage (nsys, NVTX ranges):', '', *stage_rows]
    )


def pr1(directory: Path) -> str:
    lines = []
    for label in ('A', 'OI', 'B', 'P'):
        bench, prof = directory / f'pr1_bench_{label}.json', directory / f'pr1_profile_{label}.json'
        if not bench.exists():
            continue
        b = _load(bench)
        p = _load(prof)['arms'] if prof.exists() else {}
        lines += [
            f'Set {label}: {b["images"]} files, {b["declined"]} declined by the router {b["decline_reasons"]}',
            '',
            _row(
                [
                    'arm',
                    'images/s (median of blocks)',
                    'ratio to control [95 % interval]',
                    'rep spread',
                    'host CPU s/image',
                    'GPU ms/image (kernel + copy)',
                    'gate pass/fail',
                ]
            ),
            _row(['---'] * 7),
        ]
        for spec, a in b['arms'].items():
            ratio = a.get('ratio_vs_control')
            gpu = p.get(spec)
            lines.append(
                _row(
                    [
                        spec,
                        f'{a["images_per_s_median"]:.1f}',
                        f'{ratio["ratio"]:.2f} [{ratio["lo"]:.2f}, {ratio["hi"]:.2f}]'
                        if ratio
                        else '1.00 (control)',
                        f'{a["spread"]:.2f}',
                        f'{a["cpu_s_per_image"]:.4f}',
                        f'{gpu["gpu_kernel_ms_per_image"] + gpu["gpu_memcpy_ms_per_image"]:.2f}'
                        if gpu
                        else 'n/a',
                        f'{a["parity"]["gate_pass"]}/{a["parity"]["gate_fail"]}',
                    ]
                )
            )
        cols = b['cpu_columns']
        lines += [
            '',
            'CPU columns: '
            + '; '.join(f'{k} {v["images_per_s_median"]:.1f} images/s' for k, v in cols.items()),
            '',
        ]
    return '\n'.join(lines)


def pr2(directory: Path) -> str:
    lines = []
    for label in ('A', 'B', 'P'):
        run = directory / f'pr2_run_{label}.json'
        if not run.exists():
            continue
        r = _load(run)
        gpu = [
            _load(directory / f'pr2_profile_{label}_p{i}.json')['gpu_kernel_ms_per_image']
            for i in (1, 2)
            if (directory / f'pr2_profile_{label}_p{i}.json').exists()
        ]
        lines += [
            f'Set {label}: errors {r["errors"]}',
            '',
            _row(
                [
                    'arm',
                    'operation',
                    'wall ms median / p90',
                    'GPU kernel ms/image (two passes)',
                    'gate pass/n',
                    'worst mean / max abs',
                ]
            ),
            _row(['---'] * 6),
        ]
        for arm, ops in r['wall_ms_per_op'].items():
            for op, ms in ops.items():
                if ms is None:
                    continue
                g = r['tensor_gate'][arm].get(op)
                vals = [d.get(f'{arm}/{op}') for d in gpu]
                lines.append(
                    _row(
                        [
                            arm,
                            op,
                            f'{ms:.2f} / {r["wall_ms_per_op_p90"][arm][op]:.2f}',
                            ' / '.join('n/a' if v is None else f'{v:.3f}' for v in vals) or 'n/a',
                            f'{g["passed"]}/{g["n"]}' if g else '-',
                            f'{g["mean_abs_max"]:.2f} / {g["max_abs_max"]}' if g else '-',
                        ]
                    )
                )
        lines += [
            '',
            _row(['arm', 'P-box', 'P-item', 'PE cosine median / p01', 'blur max relative diff']),
            _row(['---'] * 5),
        ]
        for arm in r['p_box']:
            pe = r['pe_cosine'][arm]
            lines.append(
                _row(
                    [
                        arm,
                        f'{r["p_box"][arm]:.4f}',
                        f'{r["p_item"][arm]:.3f}',
                        f'{pe["median"]:.5f} / {pe["p01"]:.4f}' if pe else '-',
                        r['blur_rel_diff_max'].get(arm),
                    ]
                )
            )
        lines.append('')
    return '\n'.join(lines)


def main(argv: list[str]) -> int:
    which, directory = argv[0], Path(argv[1])
    print({'pr0': pr0, 'pr1': pr1, 'pr2': pr2}[which](directory))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
