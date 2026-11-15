#!/usr/bin/env python3
"""Stamp ``item_ids`` on the images docs written before the field existed.

An images doc lists its items (``item_ids``, ``src/domain/image_completeness.py``) and a stored
image is a duplicate only when every listed item exists. A doc without the field is legacy and
counts as whole, always: this backfill never decides that a legacy photo is incomplete and never
repairs one. What it does is list the items each legacy image has now, so the audit
(``scripts/bench/ingest_consistency.py``) and every later delete cover those photos too.

It cannot find a photo an old bug stored without its objects: an image with no items looks like
an image the detector found nothing in. Both are stamped ``[]`` (a valid, complete list) and the
number of them is reported, nothing else.

Run it once per project after every API worker and job worker runs the code that writes the
field (an old worker still writing would create docs the backfill reads half written).
Resumable and idempotent: a doc that has the field is left alone, so a run that stopped
continues where it stopped. Default is a dry run; ``--apply`` writes. Each write is an OCC update
of the images doc (``if_seq_no`` / ``if_primary_term``, a 409 re-read) that sets the field only
when it is still absent.

    python3 scripts/curation/backfill_item_ids.py --project myproject            # dry run
    python3 scripts/curation/backfill_item_ids.py --project myproject --apply
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.clients.occ_bulk import occ_update_bulk
from src.config import get_curation_config
from src.db.engine import run_with_engine
from src.domain.image_completeness import ITEM_IDS_FIELD
from src.services.curation.clustering.pit_pages import iter_pages
from src.services.curation.reprocess_targets import items_by_terms
from src.services.embedding_spaces.readers import vector_source_excludes
from src.services.projects.guard import make_script_opensearch
from src.services.projects.script_binding import abind_script_project, add_project_argument


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')
MATCH_ALL: dict[str, Any] = {'match_all': {}}


@dataclass
class Counts:
    scanned: int = 0
    already_listed: int = 0
    legacy: int = 0
    stamped: int = 0
    stamped_empty: int = 0
    failed: int = 0


def _stamp(owned: dict[str, list[str]], image_id: str, source: dict[str, Any]) -> dict[str, Any]:
    """The update of one doc: its items, unless the field was written since the page was read
    (then there is nothing to do)."""
    return {} if ITEM_IDS_FIELD in source else {ITEM_IDS_FIELD: sorted(owned[image_id])}


async def backfill(
    client: Any,
    *,
    images_index: str,
    items_index: str,
    apply: bool,
    max_images: int | None = None,
) -> Counts:
    """List the items of every images doc that has no list; ``apply`` False only counts."""
    counts = Counts()
    excludes = await vector_source_excludes('images')
    async for page in iter_pages(
        client, index=images_index, query=MATCH_ALL, source=[ITEM_IDS_FIELD]
    ):
        counts.scanned += len(page)
        legacy = [h['_id'] for h in page if ITEM_IDS_FIELD not in (h.get('_source') or {})]
        counts.already_listed += len(page) - len(legacy)
        if max_images is not None:  # bounds a dry run as well: the images it would stamp
            legacy = legacy[: max(0, max_images - counts.legacy)]
        counts.legacy += len(legacy)
        if apply and legacy:
            owned: dict[str, list[str]] = {i: [] for i in legacy}
            for crop_id, source in await items_by_terms(
                client, 'image_id', legacy, index=items_index, includes=['image_id']
            ):
                owned[str(source['image_id'])].append(crop_id)
            await stamp_page(client, images_index, legacy, owned, counts, excludes)
        if max_images is not None and counts.legacy >= max_images:
            break
    return counts


async def stamp_page(
    client: Any,
    images_index: str,
    legacy: list[str],
    owned: dict[str, list[str]],
    counts: Counts,
    excludes: list[str],
) -> None:
    """One OCC bulk update of the page's legacy docs (the field is set only while absent)."""
    status = await occ_update_bulk(
        client,
        index=images_index,
        ids=legacy,
        merge_fn=functools.partial(_stamp, owned),
        vector_excludes=excludes,
    )
    for image_id in legacy:
        outcome = status.get(image_id)
        if outcome == 'updated':
            counts.stamped += 1
            counts.stamped_empty += 0 if owned[image_id] else 1
        elif outcome != 'not-found':  # deleted since the page was read: nothing to stamp
            counts.failed += 1


async def amain(args: argparse.Namespace) -> int:
    await abind_script_project(args.project)
    cfg = get_curation_config()
    client = make_script_opensearch([args.opensearch_url], use_ssl=False, timeout=300)
    try:
        counts = await backfill(
            client,
            images_index=cfg.images_index,
            items_index=cfg.items_index,
            apply=args.apply,
            max_images=args.max_images,
        )
    finally:
        await client.close()
    print(
        f'project={args.project} scanned={counts.scanned} already_listed={counts.already_listed} '
        f'legacy={counts.legacy} stamped={counts.stamped} '
        f'stamped_with_no_items={counts.stamped_empty} failed={counts.failed}'
    )
    if not args.apply:
        print(f'Dry run: {counts.legacy} images doc(s) would be stamped. Pass --apply to write.')
    return 1 if counts.failed else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--opensearch-url', default=DEFAULT_OPENSEARCH)
    p.add_argument('--apply', action='store_true', help='Write the lists (default: dry run).')
    p.add_argument('--max-images', type=int, default=None, help='Stop after this many (testing).')
    add_project_argument(p)
    return asyncio.run(run_with_engine(amain(p.parse_args(argv)), 'op-backfill-item-ids'))


if __name__ == '__main__':
    raise SystemExit(main())
