"""Seeded calibration/heldout halves of an evaluation set, split by group and never by item.

Every real evaluation set is split once. The group is the unit that must not straddle the
halves: the source image (a near-duplicate variant follows its parent), the identity for an
identity set. Anything tuned (CPM gamma, CHAMELEON floors, face thresholds) uses only the
``calibration`` half; every acceptance number is measured on ``heldout``. A manifest lists the
groups of each half, so it is stable across projects whose item ids differ, and it is hashed so
a hand edit is caught.

    eval_split.py make --labels labels.jsonl --dataset NAME --seed N --out splits/NAME.json
    eval_split.py make-pin --pin cluster_eval_3000.json --out splits/cluster_eval_3000.json

``labels.jsonl`` rows carry ``group`` (see ``cluster_eval_set.py``) and optionally ``stratum``;
groups are balanced across the halves within each stratum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

SCHEMA = 'eval-split/1'
SPLITS = ('calibration', 'heldout', 'full')


class SplitError(Exception):
    """A split manifest is inconsistent or a run would score on data it was tuned on."""


def _rank(seed: int, group: str) -> str:
    return hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()


def split_groups(
    groups: Iterable[str], seed: int, strata: Mapping[str, str] | None = None
) -> tuple[list[str], list[str]]:
    """(calibration, heldout) group lists; each stratum is halved, the odd group goes to heldout."""
    by_stratum: dict[str, list[str]] = {}
    for g in sorted(set(groups)):
        by_stratum.setdefault(strata[g] if strata else '', []).append(g)
    calibration: list[str] = []
    heldout: list[str] = []
    for members in by_stratum.values():
        ranked = sorted(members, key=lambda g: _rank(seed, g))
        calibration += ranked[: len(ranked) // 2]
        heldout += ranked[len(ranked) // 2 :]
    return sorted(calibration), sorted(heldout)


def _digest(doc: Mapping[str, Any]) -> str:
    body = {k: v for k, v in doc.items() if k != 'manifest_sha256'}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def build_manifest(
    dataset: str,
    seed: int,
    group_field: str,
    groups: Iterable[str],
    strata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    calibration, heldout = split_groups(groups, seed, strata)
    doc: dict[str, Any] = {
        'schema': SCHEMA,
        'dataset': dataset,
        'seed': seed,
        'group_field': group_field,
        'calibration': calibration,
        'heldout': heldout,
    }
    doc['manifest_sha256'] = _digest(doc)
    return doc


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')


def load_manifest(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(path.read_text())
    if doc.get('schema') != SCHEMA:
        raise SplitError(f'{path}: not an {SCHEMA} manifest')
    if doc.get('manifest_sha256') != _digest(doc):
        raise SplitError(f'{path}: manifest hash does not match its content (edited by hand?)')
    return doc


def split_groups_of(manifest: Mapping[str, Any], split: str) -> set[str]:
    if split not in SPLITS:
        raise SplitError(f'unknown split {split!r}; expected one of {SPLITS}')
    if split == 'full':
        return set(manifest['calibration']) | set(manifest['heldout'])
    return set(manifest[split])


def items_in_split(
    manifest: Mapping[str, Any], split: str, item_group: Mapping[str, str]
) -> set[str]:
    """Ids of the items whose group is in ``split``; a group the manifest never saw is an error."""
    known = split_groups_of(manifest, 'full')
    unknown = sorted({g for g in item_group.values() if g not in known})
    if unknown:
        raise SplitError(
            f'{len(unknown)} group(s) not in the split manifest, e.g. {unknown[:3]}; '
            'the labels come from different images than the split was made on'
        )
    wanted = split_groups_of(manifest, split)
    return {i for i, g in item_group.items() if g in wanted}


def assert_no_overlap(
    tuning: Mapping[str, Any], tuning_split: str, scoring: Mapping[str, Any], scoring_split: str
) -> None:
    """Refuse a run that is scored on any group it was tuned on."""
    shared = split_groups_of(tuning, tuning_split) & split_groups_of(scoring, scoring_split)
    if shared:
        raise SplitError(
            f'tuning ({tuning["dataset"]}:{tuning_split}) and scoring '
            f'({scoring["dataset"]}:{scoring_split}) overlap on {len(shared)} group(s), '
            f'e.g. {sorted(shared)[:3]}'
        )


def groups_from_pin(pin: Mapping[str, Any]) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Groups (source image names), their pin group as stratum, and variant -> parent."""
    groups = [i['file_name'] for i in pin['images']]
    strata = {i['file_name']: i['group'] for i in pin['images']}
    variant_parent = {
        v['file_name']: nd['parent']
        for nd in pin.get('near_duplicates', [])
        for v in nd['variants']
    }
    return groups, strata, variant_parent


def _cmd_make(args: argparse.Namespace) -> int:
    groups: dict[str, str] = {}
    for line in Path(args.labels).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if 'group' not in row:
                raise SystemExit(f'{args.labels}: every row needs a group (source image/identity)')
            groups.setdefault(str(row['group']), str(row.get('stratum', '')))
    manifest = build_manifest(args.dataset, args.seed, args.group_field, groups, groups)
    write_manifest(Path(args.out), manifest)
    print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in manifest.items()}))
    return 0


def _cmd_make_pin(args: argparse.Namespace) -> int:
    pin = json.loads(Path(args.pin).read_text())
    groups, strata, _ = groups_from_pin(pin)
    manifest = build_manifest(
        args.dataset or f'{Path(args.pin).stem}', args.seed, 'source_image', groups, strata
    )
    manifest['pin_manifest_sha256'] = pin['manifest_sha256']
    manifest['manifest_sha256'] = _digest(manifest)
    write_manifest(Path(args.out), manifest)
    print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in manifest.items()}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    m = sub.add_parser('make', help='split the groups of a labels.jsonl')
    m.add_argument('--labels', required=True)
    m.add_argument('--dataset', required=True)
    m.add_argument('--seed', type=int, required=True)
    m.add_argument('--group-field', default='source_image')
    m.add_argument('--out', required=True)
    m.set_defaults(func=_cmd_make)
    q = sub.add_parser('make-pin', help='split the source images of a cluster-eval pin manifest')
    q.add_argument('--pin', required=True)
    q.add_argument('--dataset')
    q.add_argument('--seed', type=int, required=True)
    q.add_argument('--out', required=True)
    q.set_defaults(func=_cmd_make_pin)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == '__main__':
    sys.exit(main())
