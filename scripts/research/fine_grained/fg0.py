"""FG-0: label check of set C by contact sheets. The reviewer sees one claimed model per sheet.

Each sheet holds the review-sample images of one claimed model plus planted controls: one image of
another model of the same make (when the make has one) and one image of another make, drawn from
outside the sample. The reviewer is told the claimed make and model, not which cells are planted.
The plants measure whether the reviewer can tell models apart at all: a reviewer that accepts
planted wrong-model images says nothing about the labels.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scripts.research.judge_stats import wilson_interval
from src.services.labeling.contact_sheet import SheetMember, build_cluster_sheet


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from scripts.research.fine_grained.setc import Item

ANSWERS = ('yes', 'no', 'unsure')
BAR_ACCURACY = 0.95
BAR_PLANT_RECALL = 0.80


class VerdictError(ValueError):
    """The reply is not a valid verdict for the sheet."""


@dataclass(frozen=True)
class SheetSpec:
    sheet_id: str
    make: str
    model: str
    sample_paths: tuple[str, ...]
    plant_paths: tuple[tuple[str, str], ...]  # (path, 'same_make' | 'other_make')


def plan_sheets(
    items: Sequence[Item], review_paths: Sequence[str], *, seed: int
) -> list[SheetSpec]:
    """One sheet per claimed model; plants come from outside the review sample."""
    by_path = {i.path: i for i in items}
    in_sample = set(review_paths)
    models = sorted({(by_path[p].make, by_path[p].model) for p in review_paths})
    pool = [i for i in items if i.path not in in_sample]
    specs = []
    for n, (make, model) in enumerate(models, 1):
        rng = random.Random(f'fg0:{seed}:{model}')
        sample = tuple(sorted(p for p in review_paths if by_path[p].model == model))
        plants: list[tuple[str, str]] = []
        same = [i for i in pool if i.make == make and i.model != model]
        other = [i for i in pool if i.make != make]
        if same:
            plants.append((rng.choice(same).path, 'same_make'))
        plants.append((rng.choice(other).path, 'other_make'))
        specs.append(SheetSpec(f'fg0_{n:02d}', make, model, sample, tuple(plants)))
    return specs


def render_prompt(spec: SheetSpec, n_cells: int, cols: int) -> str:
    return (
        f'This image is a contact sheet of {n_cells} numbered photos ({cols} columns). The photos '
        f'were filed under the label "{spec.model}" (make: {spec.make}). For each numbered cell, '
        f'decide two things from what is visible. "model": is the main vehicle a {spec.model}, of '
        f'any generation or year? "make": is the main vehicle at least a {spec.make}? Answer '
        '"yes", "no", or "unsure" when you cannot tell (a detail shot, heavy occlusion, an '
        'interior, or models you cannot tell apart). Reply with only this JSON object and nothing '
        'else: {"cells": [{"cell": 1, "model": "yes" | "no" | "unsure", "make": "yes" | "no" | '
        '"unsure"}, ...]} with exactly one entry for each cell 1 to '
        f'{n_cells}.'
    )


def build_sheet(spec: SheetSpec, root: Path, *, seed: int) -> tuple[Any, str, dict[str, str]]:
    """Render a sheet; returns ``(sheet, prompt, role_by_path)`` where role is sample or a plant."""
    roles = dict.fromkeys(spec.sample_paths, 'sample')
    roles.update(dict(spec.plant_paths))
    members = [SheetMember(item_id=p, image=(root / p).read_bytes()) for p in sorted(roles)]
    cols = 4
    rows = -(-len(members) // cols)
    sheet = build_cluster_sheet(
        members, sheet_id=spec.sheet_id, seed=seed, kind='object', grid=(cols, rows)
    )
    return sheet, render_prompt(spec, len(members), min(cols, len(members))), roles


def parse_verdict(n_cells: int, text: str) -> dict[int, dict[str, str]]:
    body = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    start, end = body.find('{'), body.rfind('}')
    if start < 0 or end < start:
        raise VerdictError('reply holds no JSON object')
    try:
        doc = json.loads(body[start : end + 1])
    except json.JSONDecodeError as exc:
        raise VerdictError(f'reply is not valid JSON: {exc.msg}') from exc
    rows = doc.get('cells')
    if not isinstance(rows, list):
        raise VerdictError('cells must be a list')
    out: dict[int, dict[str, str]] = {}
    for r in rows:
        if (
            not isinstance(r, dict)
            or r.get('cell') not in range(1, n_cells + 1)
            or r.get('model') not in ANSWERS
            or r.get('make') not in ANSWERS
        ):
            raise VerdictError('each cell needs a valid cell number, model and make answer')
        if r['cell'] in out:
            raise VerdictError(f'cell {r["cell"]} answered twice')
        out[r['cell']] = {'model': r['model'], 'make': r['make']}
    if sorted(out) != list(range(1, n_cells + 1)):
        raise VerdictError(f'every cell 1 to {n_cells} must be answered once')
    return out


def _rate(num: int, den: int) -> dict[str, Any]:
    lo, hi = wilson_interval(num, den)
    return {'n': den, 'k': num, 'rate': (num / den) if den else None, 'wilson95': [lo, hi]}


def score(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Score rows of ``{role, model, make}``; role is sample, same_make or other_make.

    Sample accuracy counts "yes" over decided rows (decided = yes or no); the strict reading counts
    "unsure" as not confirmed. Plants are wrong at model level by construction, and wrong at make
    level only when from another make. The reviewer is readable at model level only when it rejects
    at least BAR_PLANT_RECALL of the same-make plants.
    """
    sample = [r for r in rows if r['role'] == 'sample']
    same = [r for r in rows if r['role'] == 'same_make']
    other = [r for r in rows if r['role'] == 'other_make']

    def acc(level: str) -> dict[str, Any]:
        decided = [r for r in sample if r[level] != 'unsure']
        yes = sum(r[level] == 'yes' for r in decided)
        return {
            'decided': _rate(yes, len(decided)),
            'strict': _rate(sum(r[level] == 'yes' for r in sample), len(sample)),
            'unsure': sum(r[level] == 'unsure' for r in sample),
        }

    def rejects(group: Sequence[Mapping[str, str]], level: str) -> dict[str, Any]:
        return _rate(sum(r[level] == 'no' for r in group), len(group))

    plant_model = rejects(same, 'model')
    model_acc = acc('model')
    decided_rate = model_acc['decided']['rate']
    readable = plant_model['rate'] is not None and plant_model['rate'] >= BAR_PLANT_RECALL
    if not readable:
        verdict = 'INCONCLUSIVE'
    elif decided_rate is not None and decided_rate >= BAR_ACCURACY:
        verdict = 'PASS'
    else:
        verdict = 'FAIL'
    return {
        'sample_rows': len(sample),
        'model_accuracy': model_acc,
        'make_accuracy': acc('make'),
        'plants': {
            'same_make_rejected_at_model': plant_model,
            'other_make_rejected_at_model': rejects(other, 'model'),
            'other_make_rejected_at_make': rejects(other, 'make'),
        },
        'reviewer_readable_at_model_level': readable,
        'fg0_verdict': verdict,
    }
