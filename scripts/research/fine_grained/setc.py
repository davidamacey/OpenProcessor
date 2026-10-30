"""Commons set C: labels, the review sample and the registered splits. Pure apart from file reads."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class Item:
    path: str
    make: str
    model: str
    generation: str


def load_items(root: Path) -> list[Item]:
    """Every labelled image, sorted by path so the order never depends on file-system listing."""
    items = []
    for line in (root / 'labels.jsonl').read_text().splitlines():
        row = json.loads(line)
        make, model = row['taxonomy_path']
        items.append(Item(row['path'], make, model, str(row['generation'])))
    return sorted(items, key=lambda i: i.path)


def load_review_paths(root: Path) -> list[str]:
    with (root / 'review_sample.csv').open() as fh:
        return sorted(r['path'] for r in csv.DictReader(fh))


def split_halves(items: Sequence[Item], *, seed: int) -> dict[str, str]:
    """Calibration / held-out 50-50 by image, stratified by model generation, seeded."""
    by_gen: dict[tuple[str, str], list[Item]] = {}
    for it in sorted(items, key=lambda i: i.path):
        by_gen.setdefault((it.model, it.generation), []).append(it)
    out: dict[str, str] = {}
    rng = random.Random(f'half:{seed}')
    for key in sorted(by_gen):
        group = list(by_gen[key])
        rng.shuffle(group)
        cut = len(group) // 2
        # The odd image of an odd group alternates between halves so neither is favoured.
        if len(group) % 2 and rng.random() < 0.5:
            cut += 1
        for i, it in enumerate(group):
            out[it.path] = 'calibration' if i < cut else 'heldout'
    return out


def split_new_models(items: Sequence[Item], *, seed: int) -> set[str]:
    """Models held out as new: in each make with two or more models, half of them (seeded)."""
    by_make: dict[str, list[str]] = {}
    for model, make in sorted({(i.model, i.make) for i in items}):
        by_make.setdefault(make, []).append(model)
    held: set[str] = set()
    rng = random.Random(f'newmodels:{seed}')
    for make in sorted(by_make):
        models = by_make[make]
        if len(models) >= 2:
            held.update(rng.sample(models, len(models) // 2))
    return held
