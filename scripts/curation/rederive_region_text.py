#!/usr/bin/env python3
"""Re-derive each item's chosen ``region_text`` under the region-text rules.

Re-runs the text chooser on every item's *stored* readings
(``region_text_vlm`` / a VLM-sourced ``region_text`` and
``region_text_ocr``) with the active region profile's validity rules, so a
VLM reading that is not text -- the prompt's example value or a truncation
of it, a "can't read it" word, a stock run like "999" -- stops being the
chosen text and a valid OCR reading takes its place (see
``src/services/curation/region_text_repair.py``). Human-typed text is never
touched; the per-reader readings are never changed.

The rules' placeholders are the resolved prompt pack's quoted example values
(``OP_PROMPT_PACK_PATH``, or ``--prompt-pack``) plus the profile's
``text_placeholders`` plus any ``--placeholder``. Run it while the pack
still shows the example the old rows echo, or pass that example with
``--placeholder``.

Dry run by default (read-only); ``--apply`` writes under OCC.

    python3 scripts/curation/rederive_region_text.py
    python3 scripts/curation/rederive_region_text.py --placeholder ABC1234 --apply
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from opensearchpy import AsyncOpenSearch

from src.config import get_curation_config
from src.services.curation.region_text_repair import (
    apply_region_text_repair,
    plan_region_text_repair,
)
from src.services.detection.profile_registry import get_active_region_profile
from src.services.detection.region_text_rules import region_text_rules
from src.services.labeling.vlm_prompts import PromptPack, resolve_prompt_pack


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')


async def run(args: argparse.Namespace, client: object) -> int:
    profile = get_active_region_profile()
    if profile is None:
        print('No region profile configured (OP_REGION_PROFILE / OP_REGION_DETECTION_*).')
        return 2
    if args.placeholder:
        profile = dataclasses.replace(
            profile, text_placeholders=profile.text_placeholders | frozenset(args.placeholder)
        )
    pack = PromptPack.from_json(args.prompt_pack) if args.prompt_pack else resolve_prompt_pack()
    rules = region_text_rules(profile, pack)
    print(f'profile={profile.name} text_reader={profile.text_reader} pack={pack.name}')
    print(f'placeholders={sorted(rules.placeholders)}')
    plan = await plan_region_text_repair(
        client, index=args.index, profile=profile, rules=rules, page_size=args.page_size
    )
    print(
        f'{plan.scanned} item(s) with region text; {plan.human_skipped} human (untouched); '
        f'{len(plan.changes)} would change, {sum(plan.text_changed.values())} of them '
        'change the chosen text'
    )
    for name, counts in (
        ('chosen text changes', plan.text_changed),
        ('text_choice after', plan.choice),
        ('VLM reading rejected as', plan.vlm_invalid),
    ):
        print(f'  {name}:')
        for key, n in counts.most_common():
            print(f'    {key:<40} {n}')
    if args.verbose:
        for key, lines in plan.examples.items():
            print(f'  examples {key}:')
            for line in lines:
                print(f'    {line}')
    if args.dry_run:
        print('Dry-run only. Pass --apply to write.')
        return 0
    result = await apply_region_text_repair(
        client, plan, index=args.index, profile=profile, rules=rules
    )
    errors = result.get('errors') or []
    print(
        f'updated={result.get("updated", 0)} '
        f'skipped_changed={result.get("skipped_due_to_conflict", 0)} errors={len(errors)}'
    )
    return 1 if errors else 0


async def _async_main(args: argparse.Namespace) -> int:
    client = AsyncOpenSearch(hosts=[args.opensearch_url], use_ssl=False, timeout=300)
    try:
        return await run(args, client)
    finally:
        await client.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--index', default=get_curation_config().items_index)
    p.add_argument('--page-size', type=int, default=500)
    p.add_argument('--opensearch-url', default=DEFAULT_OPENSEARCH)
    p.add_argument('--prompt-pack', default=None, help='Prompt pack JSON (default: resolved).')
    p.add_argument(
        '--placeholder',
        action='append',
        default=[],
        help='Extra placeholder reading (repeatable), e.g. a retired prompt example.',
    )
    p.add_argument('--verbose', action='store_true', help='Print example rows per change.')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--dry-run', action='store_true', default=True)
    g.add_argument('--apply', dest='dry_run', action='store_false')
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.page_size <= 0:
        build_parser().error('--page-size must be positive')
    return asyncio.run(_async_main(args))


if __name__ == '__main__':
    sys.exit(main())
