"""FG-0 runner. ``prepare`` renders sheets into an exchange directory; ``collect`` scores verdicts.

Run from the repo root: ``python -m scripts.research.fine_grained.fg0_run prepare --help``.
The truth file stays outside the exchange directory so the reviewer cannot see roles.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.research.fine_grained import data, fg0, setc


def prepare(args: argparse.Namespace) -> None:
    root, out = Path(args.root), Path(args.out)
    items = setc.load_items(root)
    specs = fg0.plan_sheets(items, setc.load_review_paths(root), seed=args.seed)
    (out / 'exchange' / 'pending').mkdir(parents=True, exist_ok=True)
    (out / 'exchange' / 'verdicts').mkdir(exist_ok=True)
    truth = {}
    for spec in specs:
        sheet, prompt, roles = fg0.build_sheet(spec, root, seed=args.seed)
        (out / 'exchange' / 'pending' / f'{spec.sheet_id}.png').write_bytes(sheet.png)
        (out / 'exchange' / 'pending' / f'{spec.sheet_id}.prompt.txt').write_text(prompt)
        truth[spec.sheet_id] = {
            'make': spec.make,
            'model': spec.model,
            'png_sha256': sheet.manifest['png_sha256'],
            'cells': {
                str(c['cell']): {'path': c['item_id'], 'role': roles[c['item_id']]}
                for c in sheet.manifest['cells']
            },
        }
    (out / 'truth.json').write_text(json.dumps(truth, indent=1, sort_keys=True))
    print(f'{len(specs)} sheets in {out / "exchange" / "pending"}')


def collect(args: argparse.Namespace) -> None:
    out = Path(args.out)
    truth = json.loads((out / 'truth.json').read_text())
    rows, raw, errors = [], {}, {}
    for sheet_id, t in sorted(truth.items()):
        vpath = out / 'exchange' / 'verdicts' / f'{sheet_id}.json'
        if not vpath.exists():
            errors[sheet_id] = 'missing'
            continue
        text = vpath.read_text()
        raw[sheet_id] = text
        try:
            verdict = fg0.parse_verdict(len(t['cells']), text)
        except fg0.VerdictError as exc:
            errors[sheet_id] = str(exc)
            continue
        for cell, info in t['cells'].items():
            rows.append(
                {
                    'sheet_id': sheet_id,
                    'path': info['path'],
                    'role': info['role'],
                    **verdict[int(cell)],
                }
            )
    report = {
        'git_sha': data.git_sha(),
        'reviewer': 'machine review: Haiku subagents on contact sheets (sheet-file exchange); '
        'a person should spot check the disagreement list',
        'sheets': len(truth),
        'errors': errors,
        'score': fg0.score(rows),
        'rows': rows,
        'raw_verdicts': raw,
    }
    (out / 'fg0_report.json').write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report['score'], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in (('prepare', prepare), ('collect', collect)):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True)
        p.add_argument('--root', default=os.environ.get('FG_SETC_ROOT'))
        p.add_argument('--seed', type=int, default=0)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
