"""The registered splits of the standard benchmarks (``docs/research/wb2_standard_benchmarks.md``, section 1).

Rows come from each set's ``labels.jsonl``; the number of images and classes of every part is
checked against the registered counts, so a wrong or partial copy stops the run instead of
shifting a score.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SOP_CALIBRATION_PRODUCTS = 3000
SOP_CLUSTER_PRODUCTS = 3000
SAMPLE_SEED = 20261110


@dataclass(frozen=True)
class SetSpec:
    directory: str
    label_key: str
    split_key: str
    test_value: str
    calib_value: str
    test_images: int
    test_classes: int
    total_images: int


SETS = {
    'cars196': SetSpec('stanford_cars', 'class_id', 'dml_split', 'test', 'train', 8131, 98, 16185),
    'cub200': SetSpec('cub200_2011', 'class_id', 'dml_split', 'test', 'train', 5924, 100, 11788),
    'sop': SetSpec(
        'stanford_online_products', 'item_id', 'split', 'test', 'train', 60502, 11316, 120053
    ),
}


@dataclass(frozen=True)
class Row:
    path: str
    label: str
    part: str  # 'test' or 'calib'


class ProtocolError(RuntimeError):
    """The data on disk does not match the registered protocol."""


def sample_products(
    rows: list[dict[str, Any]], key: str, n: int, seed: int = SAMPLE_SEED
) -> set[str]:
    """``n`` distinct labels drawn with a seeded generator from the sorted label set."""
    labels = sorted({str(r[key]) for r in rows})
    return set(random.Random(seed).sample(labels, min(n, len(labels))))


def select_rows(raw: list[dict[str, Any]], spec: SetSpec, *, sop_sample: bool) -> list[Row]:
    """The evaluated ("test") and calibration ("calib") rows of one set, in file order."""
    test = [r for r in raw if r[spec.split_key] == spec.test_value]
    calib = [r for r in raw if r[spec.split_key] == spec.calib_value]
    if sop_sample:
        keep = sample_products(calib, spec.label_key, SOP_CALIBRATION_PRODUCTS)
        calib = [r for r in calib if str(r[spec.label_key]) in keep]
    key = spec.label_key
    return [Row(r['path'], str(r[key]), 'test') for r in test] + [
        Row(r['path'], str(r[key]), 'calib') for r in calib
    ]


def load_rows(data_root: Path, name: str) -> list[Row]:
    spec = SETS[name]
    path = Path(data_root) / spec.directory / 'labels.jsonl'
    raw = [json.loads(x) for x in path.read_text().splitlines() if x]
    if len(raw) != spec.total_images:
        raise ProtocolError(f'{name}: {len(raw)} labelled images, registered {spec.total_images}')
    rows = select_rows(raw, spec, sop_sample=name == 'sop')
    test = [r for r in rows if r.part == 'test']
    if len(test) != spec.test_images or len({r.label for r in test}) != spec.test_classes:
        raise ProtocolError(
            f'{name}: test part has {len(test)} images in {len({r.label for r in test})} classes, '
            f'registered {spec.test_images} in {spec.test_classes}'
        )
    return rows


def cluster_subset(rows: list[Row], name: str, part: str) -> list[int]:
    """Row indices (in ``rows``) that average linkage runs on: all of a part, or for SOP a seeded product sample."""
    idx = [i for i, r in enumerate(rows) if r.part == part]
    if name != 'sop':
        return idx
    keep = sample_products(
        [{'k': rows[i].label} for i in idx], 'k', SOP_CLUSTER_PRODUCTS, SAMPLE_SEED + 1
    )
    return [i for i in idx if rows[i].label in keep]
