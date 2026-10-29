"""Score a cluster partition against independent labels and emit JSON.

    cluster_eval.py score --labels labels.jsonl --partition partition.jsonl --out score.json \\
        [--second-partition rerun.jsonl] [--reference current.jsonl] \\
        [--runtime-s 123.4] [--peak-rss-mb 2048]

``labels.jsonl`` has one object per line: ``item_id``, ``label`` (the ground-truth class name),
optionally ``stratum`` (a hard-case group reported on its own) and ``protected`` (true for a
human-validated, import-validated, locked or holdout item). ``partition.jsonl`` has ``item_id``
and ``cluster_id`` (null or absent for an unassigned item). Any producer works: the current
pipeline, a candidate algorithm or a rerun. A labelled item missing from the partition counts as
unassigned; partition items without a label are counted and ignored.

Optional label fields: ``group`` (the source image or identity that must not straddle a split),
``view`` and ``size_bin`` (pair strata: pairs of one class that share or differ in the value) and
``identity`` (scored as its own block against the same partition, never mixed into the class
metrics). An optional ``confidence`` on a partition row adds the risk-coverage curve.

``--split-manifest M --split heldout`` scores only the items whose group is in that half of the
manifest made by ``eval_split.py``; ``--tuning-manifest T --tuning-split calibration`` states what
the producer was tuned on, and a scoring half that shares any group with it is refused.
``--scorecard-out`` writes the ``scorecard.py`` envelope (needs ``--kind --level --dataset-name
--seed``; ``--bar METRIC=VALUE`` adds the acceptance block, valid only on the heldout half).

``--second-partition`` adds the stability of two runs. ``--reference`` adds how many protected
items the partition moves relative to the current one. ``--runtime-s`` and ``--peak-rss-mb`` are
the producer's own cost, recorded as given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench import cluster_eval_metrics as metrics, eval_split, scorecard


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise SystemExit(f'{path}:{n}: not JSON ({exc})') from exc
            if not isinstance(row, dict) or 'item_id' not in row:
                raise SystemExit(f'{path}:{n}: expected an object with item_id')
            rows.append(row)
    return rows


OPTIONAL_LABEL_FIELDS = ('group', 'view', 'size_bin', 'identity')


def read_labels(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Item ids in file order and aligned ``truth``, ``stratum`` and ``protected`` arrays.

    An optional field (``group``, ``view``, ``size_bin``, ``identity``) must be on every row or
    on none; it is returned only when present.
    """
    rows = _jsonl(path)
    ids = [str(r['item_id']) for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f'{path}: duplicate item_id')
    if any('label' not in r for r in rows):
        raise SystemExit(f'{path}: every row needs a label')
    out = {
        'truth': np.array([str(r['label']) for r in rows]),
        'stratum': np.array([str(r.get('stratum', 'all')) for r in rows]),
        'protected': np.array([bool(r.get('protected', False)) for r in rows], dtype=bool),
    }
    for field in OPTIONAL_LABEL_FIELDS:
        have = sum(field in r for r in rows)
        if have not in (0, len(rows)):
            raise SystemExit(f'{path}: {field} is on {have} of {len(rows)} rows; all or none')
        if have:
            out[field] = np.array([str(r[field]) for r in rows])
    return ids, out


def read_partition(path: Path, ids: Sequence[str]) -> tuple[np.ndarray, int]:
    """Cluster codes aligned to ``ids`` (unassigned = -1) and the count of unlabelled rows."""
    code: dict[str, int] = {}
    by_item: dict[str, int] = {}
    for r in _jsonl(path):
        item = str(r['item_id'])
        if item in by_item:
            raise SystemExit(f'{path}: duplicate item_id {item}')
        cid = r.get('cluster_id')
        by_item[item] = metrics.UNASSIGNED if cid is None else code.setdefault(str(cid), len(code))
    known = set(ids)
    extra = sum(1 for item in by_item if item not in known)
    return np.array([by_item.get(i, metrics.UNASSIGNED) for i in ids], dtype=np.int64), extra


def read_confidence(path: Path, ids: Sequence[str]) -> np.ndarray | None:
    """Per-item ``confidence`` of the partition rows aligned to ``ids``, or None if absent."""
    rows = {str(r['item_id']): r.get('confidence') for r in _jsonl(path)}
    if all(v is None for v in rows.values()):
        return None
    return np.array([float(rows.get(i) or 0.0) for i in ids])


def _class_block(truth: np.ndarray, cluster: np.ndarray) -> dict[str, Any]:
    return {
        **metrics.pairwise_scores(truth, cluster),
        **metrics.bcubed_scores(truth, cluster),
        'ami': metrics.adjusted_mutual_info(truth, cluster),
    }


def _select_split(
    args: argparse.Namespace, ids: Sequence[str], lab: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, Any] | None, dict[str, Any] | None]:
    """Mask of the scored items, the scoring manifest and the tuning manifest (if given)."""
    if not args.split_manifest:
        if args.split or args.tuning_manifest:
            raise SystemExit('--split and --tuning-manifest need --split-manifest')
        return np.ones(len(ids), dtype=bool), None, None
    args.split = args.split or 'heldout'
    args.tuning_split = args.tuning_split or 'calibration'
    if 'group' not in lab:
        raise SystemExit('splitting needs a group on every label row (source image or identity)')
    manifest = eval_split.load_manifest(Path(args.split_manifest))
    try:
        chosen = eval_split.items_in_split(
            manifest, args.split, dict(zip(ids, lab['group'], strict=True))
        )
        tuning = None
        if args.tuning_manifest:
            tuning = eval_split.load_manifest(Path(args.tuning_manifest))
            eval_split.assert_no_overlap(tuning, args.tuning_split, manifest, args.split)
    except eval_split.SplitError as exc:
        raise SystemExit(str(exc)) from exc
    return np.array([i in chosen for i in ids], dtype=bool), manifest, tuning


def score(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    ids, lab = read_labels(Path(args.labels))
    cluster, extra = read_partition(Path(args.partition), ids)
    keep, manifest, tuning = _select_split(args, ids, lab)
    cluster = cluster[keep]
    lab = {k: v[keep] for k, v in lab.items()}
    out: dict[str, Any] = {
        'partition': args.partition,
        'labels': args.labels,
        'partition_rows_without_label': extra,
        'classes': len(np.unique(lab['truth'])),
        **_class_block(lab['truth'], cluster),
    }
    if manifest:
        out['split'] = {'name': args.split, 'manifest_sha256': manifest['manifest_sha256']}
    if tuning:
        out['tuning'] = {'split': args.tuning_split, 'manifest_sha256': tuning['manifest_sha256']}
    confidence = read_confidence(Path(args.partition), ids)
    out['risk_coverage'] = metrics.risk_coverage(
        lab['truth'], cluster, None if confidence is None else confidence[keep]
    )
    if len(np.unique(lab['stratum'])) > 1:
        out['strata'] = metrics.stratified_scores(lab['truth'], cluster, lab['stratum'])
    for field, block in (('view', 'view_pairs'), ('size_bin', 'size_pairs')):
        if field in lab:
            out[block] = metrics.attribute_pair_scores(lab['truth'], cluster, lab[field])
    if 'identity' in lab:
        out['identity'] = _class_block(lab['identity'], cluster)
    if args.second_partition:
        second, _ = read_partition(Path(args.second_partition), ids)
        out['stability'] = metrics.assignment_stability(cluster, second[keep])
    if args.reference:
        ref, _ = read_partition(Path(args.reference), ids)
        out['protected_items'] = int(lab['protected'].sum())
        out['protected_items_moved'] = metrics.moved_items(ref[keep], cluster, lab['protected'])
    out['producer_runtime_s'] = args.runtime_s
    out['producer_peak_rss_mb'] = args.peak_rss_mb
    out['scoring_s'] = time.perf_counter() - started
    return out


def make_scorecard(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    """The ``scorecard.py`` envelope of a ``score`` result."""
    split = result.get('split')
    labels_hash = hashlib.sha256(Path(args.labels).read_bytes()).hexdigest()
    dataset: dict[str, Any] = {
        'name': args.dataset_name,
        'synthetic': False,
        'split': split['name'] if split else 'full',
        'manifest_sha256': split['manifest_sha256'] if split else labels_hash,
    }
    if 'tuning' in result:
        dataset['tuning'] = [result['tuning']]
    skip = {'partition', 'labels', 'strata', 'tuning', 'split'}
    flat = {
        k: v
        for k, v in result.items()
        if k not in skip and (v is None or isinstance(v, bool | int | float))
    }
    nested = {k: v for k, v in result.items() if isinstance(v, dict) and k not in skip}
    bar = None
    if args.bar:
        name, _, value = args.bar.partition('=')
        if name not in flat:
            raise SystemExit(f'--bar names {name!r}, which is not a scalar metric of this run')
        bar = scorecard.acceptance(name, float(value), flat[name])
    meta = {
        'git_sha': args.git_sha or scorecard.current_git_sha(),
        'seed': args.seed,
        'dataset_manifest_sha256': dataset['manifest_sha256'],
        'embedding_field': args.embedding_field,
        'producer_runtime_s': args.runtime_s,
        'producer_peak_rss_mb': args.peak_rss_mb,
    }
    return scorecard.build(
        args.kind,
        args.level,
        dataset,
        meta,
        {**flat, **nested},
        result.get('strata'),
        bar,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('score', help='score one partition and write JSON')
    s.add_argument('--labels', required=True)
    s.add_argument('--partition', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--second-partition')
    s.add_argument('--reference')
    s.add_argument('--runtime-s', type=float)
    s.add_argument('--peak-rss-mb', type=float)
    s.add_argument('--split-manifest', help='eval_split.py manifest; needs a group on each label')
    s.add_argument('--split', choices=eval_split.SPLITS)
    s.add_argument('--tuning-manifest', help='manifest of what the producer was tuned on')
    s.add_argument('--tuning-split', choices=eval_split.SPLITS)
    s.add_argument('--scorecard-out', help='write the scorecard envelope here')
    s.add_argument('--kind', help='scorecard kind, e.g. K2')
    s.add_argument('--level', help='scorecard level, e.g. items')
    s.add_argument('--dataset-name')
    s.add_argument('--seed', type=int)
    s.add_argument('--git-sha')
    s.add_argument('--embedding-field')
    s.add_argument('--bar', help='METRIC=VALUE acceptance bar (inclusive), heldout only')
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scorecard_out and None in (args.kind, args.level, args.dataset_name, args.seed):
        raise SystemExit('--scorecard-out needs --kind, --level, --dataset-name and --seed')
    result = score(args)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    if args.scorecard_out:
        scorecard.write(Path(args.scorecard_out), make_scorecard(args, result))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
