#!/usr/bin/env python3
"""Score the VLM detector hint on/off against an oracle (#61 item 2).

Two steps, run by hand around two labeling runs of the same crops (one with a
pack whose ``detector_hint_min_confidence_pct`` is 0, one with it set):

    # after each run, dump (crop, oracle name, VLM answer) for that run's pack
    .venv/bin/python scripts/curation/bakeoff/vlm_hint_oracle.py dump \\
        --project <slug> --pack <pack-name> --truth-field <field> --out off.jsonl

    .venv/bin/python scripts/curation/bakeoff/vlm_hint_oracle.py compare \\
        --off off.jsonl --on on.jsonl

``dump`` reads items whose ``vlm_prompt_pack`` stamp starts with ``--pack`` and
writes ``{"crop_id", "truth", "answer"}`` per line; ``answer`` is the registry
class name the VLM landed on, or ``""`` when it gave none (unmatched, new class,
empty). ``compare`` reports overall accuracy (an unanswered crop counts as wrong)
and accuracy when answered, for each run, over the crops both runs cover.
Class identity is the name throughout. Read-only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def score(records: list[dict[str, str]]) -> dict[str, float | int]:
    """Overall accuracy, accuracy when answered, and the answered fraction."""
    total = len(records)
    answered = [r for r in records if r['answer']]
    correct = sum(1 for r in answered if r['answer'] == r['truth'])
    return {
        'n': total,
        'answered': len(answered),
        'overall_accuracy': correct / total if total else 0.0,
        'answered_accuracy': correct / len(answered) if answered else 0.0,
    }


def load_records(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[str(rec['crop_id'])] = {
            'truth': str(rec['truth']),
            'answer': str(rec.get('answer') or ''),
        }
    return out


def compare(off: dict[str, dict[str, str]], on: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Score both runs over the crops present in both; refuse when none are shared."""
    shared = sorted(off.keys() & on.keys())
    if not shared:
        raise ValueError('the two runs share no crop ids')
    mismatched = [c for c in shared if off[c]['truth'] != on[c]['truth']]
    if mismatched:
        raise ValueError(f'oracle names differ between runs for {len(mismatched)} crop(s)')
    off_s = score([off[c] for c in shared])
    on_s = score([on[c] for c in shared])
    return {
        'off': off_s,
        'on': on_s,
        'overall_accuracy_delta': float(on_s['overall_accuracy'])
        - float(off_s['overall_accuracy']),
        'dropped_not_in_both': len(off.keys() ^ on.keys()),
    }


async def _dump(args: argparse.Namespace) -> int:
    from src.config import get_curation_config
    from src.services.projects.guard import make_script_opensearch
    from src.services.projects.script_binding import abind_script_project

    await abind_script_project(args.project, opensearch_url=args.opensearch_url)
    client = make_script_opensearch([args.opensearch_url], use_ssl=False, timeout=300)
    index = get_curation_config().items_index
    written = 0
    try:
        search_after: list[Any] | None = None
        with args.out.open('w') as fh:
            while True:
                body: dict[str, Any] = {
                    'size': 500,
                    'sort': [{'crop_id': 'asc'}],
                    '_source': ['crop_id', 'class_name', 'class_source', args.truth_field],
                    'query': {'bool': {'filter': [{'prefix': {'vlm_prompt_pack': args.pack}}]}},
                }
                if search_after:
                    body['search_after'] = search_after
                hits = (await client.search(index=index, body=body))['hits']['hits']
                if not hits:
                    break
                for hit in hits:
                    src = hit['_source']
                    truth = src.get(args.truth_field)
                    if not truth:
                        continue
                    answered = src.get('class_source') == 'vlm' and src.get('class_name')
                    rec = {
                        'crop_id': src['crop_id'],
                        'truth': truth,
                        'answer': src['class_name'] if answered else '',
                    }
                    fh.write(json.dumps(rec) + '\n')
                    written += 1
                search_after = hits[-1]['sort']
    finally:
        await client.close()
    print(f'wrote {written} record(s) to {args.out}')
    return 0 if written else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('dump', help='write (crop, truth, answer) records for one run')
    d.add_argument('--project', required=True)
    d.add_argument('--pack', required=True, help='pack name the run used (stamp prefix)')
    d.add_argument('--truth-field', required=True, help='item field holding the oracle name')
    d.add_argument('--out', type=Path, required=True)
    d.add_argument('--opensearch-url', default='http://localhost:4607')
    c = sub.add_parser('compare', help='score an off run against an on run')
    c.add_argument('--off', type=Path, required=True)
    c.add_argument('--on', type=Path, required=True)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == 'dump':
        return asyncio.run(_dump(args))
    try:
        result = compare(load_records(args.off), load_records(args.on))
    except ValueError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
