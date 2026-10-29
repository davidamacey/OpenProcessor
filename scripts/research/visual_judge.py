#!/usr/bin/env python3
"""Visual judge: build contact-sheet samples, run a judge over them, verify they rebuild.

    visual_judge.py make-e0 --pool P/pool.jsonl --kind object --level "object class" --out DIR
    visual_judge.py make-calibration --pool ... --out DIR       (same clusters at 5 layouts)
    visual_judge.py make-clusters --clusters c.json --out DIR   (real clusters, 50/50 split)
    visual_judge.py make-search --spec s.json --out DIR
    visual_judge.py run --sheets DIR --backend local|sheet-file ...
    visual_judge.py verify --sheets DIR --pool P/pool.jsonl

Backends. ``local``: a local OpenAI-compatible VLM (this repo's ``vlm`` service, default
http://127.0.0.1:4612/v1); a non-local endpoint is refused. ``sheet-file``: a hosted judge without
an API key in this process. ``run`` writes ``EXCHANGE/pending/<id>.png`` and ``<id>.prompt.txt``;
a judge that can view image files (a Haiku subagent) writes ``EXCHANGE/verdicts/<id>.json`` with
the JSON the prompt asks for; running the same command again reads the verdicts into the JSONL
record. Face, identity-text and plate sheets are refused for any hosted backend. Every run
reserves its call budget on disk before each call (``--max-calls``, ``--max-input-tokens``).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.scorecard import current_git_sha as _git_sha
from scripts.research import judge_cases as jc, judge_sheetdir as sd
from scripts.research.judge_backends import LocalVlmBackend, SheetFileBackend
from scripts.research.judge_run import Budget, VerdictCache, run_requests
from src.services.labeling import contact_sheet as cs


if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_ENDPOINT = 'http://127.0.0.1:4612/v1'
CLEAN_FRACTION = 0.45
BAD_COUNTS = (1, 2, 3, 5, 8)
CONTROL_FRACTION = 0.1
CONTROL_BAD = 8
CALIBRATION_LAYOUTS = ((3, 3, 160), (4, 4, 160), (6, 4, 160), (6, 4, 128), (6, 4, 192))


def _spec(args: argparse.Namespace, **extra: Any) -> dict[str, Any]:
    return {
        'kind': args.kind,
        'level': args.level,
        'seed': args.seed,
        'builder_version': cs.BUILDER_VERSION,
        'pool': str(args.pool),
        'git_sha': _git_sha(),
        **extra,
    }


def _only(args: argparse.Namespace) -> set[str] | None:
    return set(args.labels.split(',')) if args.labels else None


def _bad_count(rng: random.Random) -> int:
    return 0 if rng.random() < CLEAN_FRACTION else rng.choice(BAD_COUNTS)


def make_e0(args: argparse.Namespace) -> None:
    pool = jc.load_pool(Path(args.pool), _only(args))
    rng = random.Random(f'{args.seed}:e0')
    out = Path(args.out)
    labels: dict[str, str] = {}
    truth: list[dict[str, Any]] = []
    for k in range(args.n_cluster):
        case = jc.synthetic_cluster(
            pool, rng, case_id=f'cl{k:04d}', size=args.size, n_bad=_bad_count(rng)
        )
        labels.update(case.labels)
        variants = [(f'cl{k:04d}', 0)]
        if rng.random() < args.reorder_frac:
            variants.append((f'cl{k:04d}-r', 1))
        for sheet_id, variant in variants:
            sheet = cs.build_cluster_sheet(
                case.members,
                sheet_id=sheet_id,
                seed=args.seed,
                kind=args.kind,
                grid=(args.cols, args.rows),
                thumb_px=args.thumb,
                selection_id=f'cl{k:04d}',
                order_variant=variant,
            )
            sd.write_sheet(out, sheet)
            truth.append(
                {
                    'sheet_id': sheet_id,
                    'case': case.case_id,
                    'layout': 'cluster',
                    'n_bad': case.n_bad,
                    'bad_kind': case.bad_kind,
                    'truth': jc.sheet_truth(sheet.manifest, case.labels),
                }
            )
    for k in range(args.n_pair):
        pairs, pair_labels = jc.pair_rows(pool, rng)
        labels.update(pair_labels)
        sheet = cs.build_pair_sheet(pairs, sheet_id=f'pr{k:04d}', seed=args.seed, kind=args.kind)
        sd.write_sheet(out, sheet)
        truth.append(
            {
                'sheet_id': sheet.sheet_id,
                'case': sheet.sheet_id,
                'layout': 'pair',
                'truth': jc.sheet_truth(sheet.manifest, pair_labels),
            }
        )
    sd.write_meta(out, _spec(args, n_cluster=args.n_cluster, n_pair=args.n_pair), labels, truth)
    print(f'{len(truth)} sheets in {out}')


def make_calibration(args: argparse.Namespace) -> None:
    pool = jc.load_pool(Path(args.pool), _only(args))
    rng = random.Random(f'{args.seed}:cal')
    out = Path(args.out)
    labels: dict[str, str] = {}
    truth: list[dict[str, Any]] = []
    for k in range(args.n_cluster):
        case = jc.synthetic_cluster(
            pool, rng, case_id=f'cal{k:03d}', size=args.size, n_bad=rng.choice((0, 2, 3, 5, 8))
        )
        labels.update(case.labels)
        for cols, rows, thumb in CALIBRATION_LAYOUTS:
            sheet_id = f'cal{k:03d}-{cols}x{rows}-{thumb}'
            sheet = cs.build_cluster_sheet(
                case.members,
                sheet_id=sheet_id,
                seed=args.seed,
                kind=args.kind,
                grid=(cols, rows),
                thumb_px=thumb,
                selection_id=case.case_id,
            )
            sd.write_sheet(out, sheet)
            truth.append(
                {
                    'sheet_id': sheet_id,
                    'case': case.case_id,
                    'layout': 'cluster',
                    'grid': f'{cols}x{rows}',
                    'thumb_px': thumb,
                    'n_bad': case.n_bad,
                    'truth': jc.sheet_truth(sheet.manifest, case.labels),
                }
            )
    sd.write_meta(out, _spec(args, calibration=True), labels, truth)
    print(f'{len(truth)} sheets in {out}')


def _members(rows: Sequence[dict[str, Any]]) -> list[cs.SheetMember]:
    return [
        cs.SheetMember(
            item_id=r['item_id'],
            image=Path(r['path']).read_bytes(),
            box_id=r.get('box_id', ''),
            bbox_norm=tuple(r['bbox_norm']) if r.get('bbox_norm') else None,
            is_prototype=bool(r.get('is_prototype')),
            distance=r.get('distance'),
        )
        for r in rows
    ]


def make_clusters(args: argparse.Namespace) -> None:
    """Real clusters: ``{"clusters": {id: {"tags": [...], "members": [...]}}}``; half random."""
    doc = json.loads(Path(args.clusters).read_text())['clusters']
    tags = {cid: set(c.get('tags', [])) for cid, c in doc.items()}
    picks = jc.targeted_random_split(tags, args.n_cluster, args.seed)
    out = Path(args.out)
    truth: list[dict[str, Any]] = []
    for n, (cid, sample) in enumerate(picks):
        sheet = cs.build_cluster_sheet(
            _members(doc[cid]['members']),
            sheet_id=f'rc{n:04d}',
            seed=args.seed,
            kind=args.kind,
            grid=(args.cols, args.rows),
            thumb_px=args.thumb,
        )
        sd.write_sheet(out, sheet)
        truth.append(
            {
                'sheet_id': sheet.sheet_id,
                'case': cid,
                'layout': 'cluster',
                'sample': sample,
                'tags': sorted(tags[cid]),
                'truth': None,
            }
        )
    labels = _plant_controls(args, out, len(picks), truth) if args.controls_pool else {}
    sd.write_meta(out, _spec(args), labels, truth)
    print(f'{len(truth)} cluster sheets in {out} ({len(picks)} sampled)')


def _plant_controls(
    args: argparse.Namespace, out: Path, n_sampled: int, truth: list[dict[str, Any]]
) -> dict[str, str]:
    """About 10 % known-same and known-different sheets from labelled public data."""
    pool = jc.load_pool(Path(args.controls_pool))
    rng = random.Random(f'{args.seed}:controls')
    labels: dict[str, str] = {}
    for k in range(max(2, round(CONTROL_FRACTION * n_sampled))):
        same = k % 2 == 0
        case = jc.synthetic_cluster(
            pool,
            rng,
            case_id=f'ctl{k:03d}',
            size=args.size,
            n_bad=0 if same else CONTROL_BAD,
            hard_fraction=0.0,
        )
        labels.update(case.labels)
        sheet = cs.build_cluster_sheet(
            case.members,
            sheet_id=f'rc-ctl{k:03d}',
            seed=args.seed,
            kind=args.kind,
            grid=(args.cols, args.rows),
            thumb_px=args.thumb,
        )
        sd.write_sheet(out, sheet)
        truth.append(
            {
                'sheet_id': sheet.sheet_id,
                'case': case.case_id,
                'layout': 'cluster',
                'control': 'same' if same else 'different',
                'sample': 'control',
                'truth': jc.sheet_truth(sheet.manifest, case.labels),
            }
        )
    return labels


def make_search(args: argparse.Namespace) -> None:
    """``{"queries": [{"id", "query": {...}, "results": [{...}, ...]}]}`` (members as in clusters)."""
    queries = json.loads(Path(args.spec).read_text())['queries']
    out = Path(args.out)
    truth = []
    for q in queries:
        sheet = cs.build_search_sheet(
            _members([q['query']])[0],
            _members(q['results']),
            sheet_id=f'sq-{q["id"]}',
            seed=args.seed,
            kind=args.kind,
            thumb_px=args.thumb,
        )
        sd.write_sheet(out, sheet)
        truth.append(
            {'sheet_id': sheet.sheet_id, 'case': q['id'], 'layout': 'search', 'truth': None}
        )
    sd.write_meta(out, _spec(args), {}, truth)
    print(f'{len(truth)} search sheets in {out}')


def verify(args: argparse.Namespace) -> None:
    """Rebuild every sheet from its manifest and the pool crops; any mismatch is an error."""
    crops = {
        cs.member_key(i.item_id, ''): Path(i.path).read_bytes()
        for items in jc.load_pool(Path(args.pool)).values()
        for i in items
    }
    bad = 0
    for sheet_id, manifest in sd.load_manifests(Path(args.sheets)).items():
        try:
            png = cs.rebuild_sheet(manifest, crops).png
            if png != (Path(args.sheets) / 'sheets' / f'{sheet_id}.png').read_bytes():
                raise cs.SheetRebuildError('stored PNG differs from the rebuilt one')
        except cs.SheetRebuildError as exc:
            bad += 1
            print(f'{sheet_id}: {exc}')
    print(f'verified, {bad} mismatches')
    if bad:
        raise SystemExit(1)


def run(args: argparse.Namespace) -> None:
    sheets = Path(args.sheets)
    spec = sd.load_spec(sheets)
    if args.backend == 'local':
        backend: LocalVlmBackend | SheetFileBackend = LocalVlmBackend(
            args.endpoint, model=args.model, max_tokens=args.max_output_tokens
        )
    else:
        backend = SheetFileBackend(Path(args.exchange_dir), model=args.model)
    requests = sd.load_requests(sheets, level=spec['level'], prefix=args.prefix, limit=args.limit)
    run_id = (
        args.run_id
        or f'{backend.name}-{backend.model.replace("/", "_")}-{time.strftime("%Y%m%d%H%M%S")}'
    )
    out = Path(args.out) if args.out else sheets / 'judge' / f'{run_id}.jsonl'
    budget = Budget(
        Path(
            args.budget_file
            or sheets / 'judge' / f'budget-{backend.name}-{backend.model.replace("/", "_")}.json'
        ),
        max_calls=args.max_calls,
        max_input_tokens=args.max_input_tokens,
    )
    cache = VerdictCache(Path(args.cache)) if args.cache else None
    outcome = run_requests(
        requests,
        backend,
        budget,
        cache,
        out,
        run_id=run_id,
        header={
            'seed': spec['seed'],
            'git_sha': _git_sha(),
            'kind': spec['kind'],
            'dataset_pool': spec['pool'],
            'sheets_dir': str(sheets),
        },
    )
    print(
        f'{run_id}: judged {len(outcome.judged)}, cached {len(outcome.cached)}, '
        f'pending {len(outcome.pending)}, invalid {len(outcome.invalid)} -> {out}'
    )


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument('--pool', required=True)
    p.add_argument('--kind', required=True, help='object, vehicle, face, plate, ...')
    p.add_argument('--level', required=True, help='what "same" means, e.g. "object class"')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cols', type=int, default=cs.DEFAULT_GRID[0])
    p.add_argument('--rows', type=int, default=cs.DEFAULT_GRID[1])
    p.add_argument('--thumb', type=int, default=cs.DEFAULT_THUMB_PX)
    p.add_argument('--size', type=int, default=24, help='members per synthetic cluster')
    p.add_argument('--labels', help='comma-separated labels to draw from (default: all)')


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e0 = sub.add_parser('make-e0')
    _common(e0)
    e0.add_argument('--n-cluster', type=int, default=200)
    e0.add_argument('--n-pair', type=int, default=42)
    e0.add_argument('--reorder-frac', type=float, default=0.1)
    e0.set_defaults(fn=make_e0)
    cal = sub.add_parser('make-calibration')
    _common(cal)
    cal.add_argument('--n-cluster', type=int, default=30)
    cal.set_defaults(fn=make_calibration)
    cl = sub.add_parser('make-clusters')
    cl.add_argument('--clusters', required=True)
    cl.add_argument('--pool', default='')
    cl.add_argument('--controls-pool', help='labelled public pool for planted controls')
    cl.add_argument('--size', type=int, default=24)
    cl.add_argument('--kind', required=True)
    cl.add_argument('--level', required=True)
    cl.add_argument('--out', required=True)
    cl.add_argument('--n-cluster', type=int, default=20)
    cl.add_argument('--seed', type=int, default=0)
    cl.add_argument('--cols', type=int, default=cs.DEFAULT_GRID[0])
    cl.add_argument('--rows', type=int, default=cs.DEFAULT_GRID[1])
    cl.add_argument('--thumb', type=int, default=cs.DEFAULT_THUMB_PX)
    cl.set_defaults(fn=make_clusters)
    se = sub.add_parser('make-search')
    se.add_argument('--spec', required=True)
    se.add_argument('--pool', default='')
    se.add_argument('--kind', required=True)
    se.add_argument('--level', required=True)
    se.add_argument('--out', required=True)
    se.add_argument('--seed', type=int, default=0)
    se.add_argument('--thumb', type=int, default=cs.DEFAULT_THUMB_PX)
    se.set_defaults(fn=make_search)
    ve = sub.add_parser('verify')
    ve.add_argument('--sheets', required=True)
    ve.add_argument('--pool', required=True)
    ve.set_defaults(fn=verify)
    rn = sub.add_parser('run')
    rn.add_argument('--sheets', required=True)
    rn.add_argument('--backend', choices=('local', 'sheet-file'), required=True)
    rn.add_argument('--model', required=True, help='served model name, or a label for sheet-file')
    rn.add_argument('--endpoint', default=DEFAULT_ENDPOINT)
    rn.add_argument('--exchange-dir')
    rn.add_argument('--prefix', default='')
    rn.add_argument('--limit', type=int)
    rn.add_argument('--run-id')
    rn.add_argument('--out')
    rn.add_argument('--cache')
    rn.add_argument('--budget-file')
    rn.add_argument('--max-calls', type=int, required=True)
    rn.add_argument('--max-input-tokens', type=int, required=True)
    rn.add_argument('--max-output-tokens', type=int, default=600)
    rn.set_defaults(fn=run)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == 'run' and args.backend == 'sheet-file' and not args.exchange_dir:
        build_parser().error('--exchange-dir is required for the sheet-file backend')
    args.fn(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
