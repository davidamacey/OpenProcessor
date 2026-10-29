"""On-disk layout of a judge sample: sheets, manifests, labels and the truth that stays beside them.

DIR/spec.json            level, kind, seed, pool, builder version
DIR/sheets/<id>.png      what a judge sees
DIR/manifests/<id>.json  cell -> item mapping and png_sha256
DIR/labels.json          item key -> true label (never shown to a judge)
DIR/truth.jsonl          one row per sheet: layout, case, truth
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from scripts.research.judge_protocol import JudgeRequest, render_prompt


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from src.services.labeling.contact_sheet import Sheet


def write_sheet(directory: Path, sheet: Sheet) -> None:
    (directory / 'sheets').mkdir(parents=True, exist_ok=True)
    (directory / 'manifests').mkdir(exist_ok=True)
    (directory / 'sheets' / f'{sheet.sheet_id}.png').write_bytes(sheet.png)
    (directory / 'manifests' / f'{sheet.sheet_id}.json').write_text(
        json.dumps(sheet.manifest, indent=1, sort_keys=True)
    )


def write_meta(
    directory: Path,
    spec: Mapping[str, Any],
    labels: Mapping[str, str],
    truth: Iterable[Mapping[str, Any]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'spec.json').write_text(json.dumps(spec, indent=1, sort_keys=True))
    (directory / 'labels.json').write_text(json.dumps(labels, sort_keys=True))
    with (directory / 'truth.jsonl').open('w') as fh:
        for row in truth:
            fh.write(json.dumps(row, sort_keys=True) + '\n')


def load_spec(directory: Path) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads((directory / 'spec.json').read_text())
    return spec


def load_truth(directory: Path) -> dict[str, dict[str, Any]]:
    rows = (json.loads(x) for x in (directory / 'truth.jsonl').read_text().splitlines())
    return {r['sheet_id']: r for r in rows}


def load_manifests(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        p.stem: json.loads(p.read_text()) for p in sorted((directory / 'manifests').glob('*.json'))
    }


def load_requests(
    directory: Path, *, level: str, prefix: str = '', limit: int | None = None
) -> list[JudgeRequest]:
    out: list[JudgeRequest] = []
    for sheet_id, manifest in load_manifests(directory).items():
        if not sheet_id.startswith(prefix):
            continue
        png = (directory / 'sheets' / f'{sheet_id}.png').read_bytes()
        out.append(JudgeRequest(manifest, png, render_prompt(manifest, level=level)))
    return out[:limit]
