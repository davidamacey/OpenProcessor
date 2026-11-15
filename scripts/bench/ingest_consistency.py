#!/usr/bin/env python3
"""Is every photo of a project stored whole? A read-only audit; exit code 1 on any inconsistency.

The rule (``src/domain/image_completeness.py``): an images doc lists its items in ``item_ids`` and
every listed item must exist. This script reads the bound project's indexes and its exact-content
rows and counts:

* ``images_missing_items``: images docs with a listed item that does not exist (the photo was
  stored without its objects: a kill or a failed write that nobody repaired);
* ``items_without_image``: items whose images doc does not exist;
* ``contents_without_image``: exact-content rows (``image_contents``) whose images doc does not
  exist (a claim that nothing wrote; the next lookup of the bytes releases it, but a sweep ends
  with none).

``images_without_list`` (legacy docs, written before the field or copied by a combine) and
``images_with_empty_list`` (a photo with no detections) are reported as numbers only: they are not
inconsistencies. Run it after every kill, outage and fault-injection test of the ingest sweep;
the bar is 0 inconsistencies. Nothing is written.

    ingest_consistency.py --project opbench --opensearch-url http://127.0.0.1:4607
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import get_curation_config
from src.db.engine import run_with_engine
from src.domain.image_completeness import ITEM_IDS_FIELD, listed_item_ids
from src.services.curation.clustering.pit_pages import iter_pages
from src.services.curation.image_completeness import missing_items
from src.services.projects.guard import make_script_opensearch
from src.services.projects.script_binding import abind_script_project, add_project_argument


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

EXAMPLES = 5
MATCH_ALL: dict[str, Any] = {'match_all': {}}
ROWS_PAGE = 5000


@dataclass
class Report:
    images: int = 0
    items: int = 0
    content_rows: int = 0
    images_missing_items: int = 0
    items_without_image: int = 0
    contents_without_image: int = 0
    images_without_list: int = 0
    images_with_empty_list: int = 0
    examples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def inconsistencies(self) -> int:
        return self.images_missing_items + self.items_without_image + self.contents_without_image

    def note(self, kind: str, example: str) -> None:
        shown = self.examples.setdefault(kind, [])
        if len(shown) < EXAMPLES:
            shown.append(example)


async def absent_images(client: Any, images_index: str, image_ids: Sequence[str]) -> list[str]:
    """The ids of ``image_ids`` that have no images doc (one real-time ``mget`` per page)."""
    wanted = sorted(set(image_ids))
    found: set[str] = set()
    for start in range(0, len(wanted), 1000):
        resp = await client.mget(
            index=images_index, body={'ids': wanted[start : start + 1000]}, _source=False
        )
        found.update(d['_id'] for d in resp.get('docs') or [] if d.get('found'))
    return [i for i in wanted if i not in found]


async def check_images(client: Any, report: Report, *, images_index: str, items_index: str) -> None:
    """Every listed item of every images doc exists."""
    async for page in iter_pages(
        client, index=images_index, query=MATCH_ALL, source=[ITEM_IDS_FIELD]
    ):
        listed: dict[str, list[str]] = {}
        for hit in page:
            report.images += 1
            ids = listed_item_ids(hit.get('_source') or {})
            if ids is None:
                report.images_without_list += 1
            elif not ids:
                report.images_with_empty_list += 1
            else:
                listed[hit['_id']] = ids
        for image_id in await missing_items(client, items_index, listed):
            report.images_missing_items += 1
            report.note('images_missing_items', image_id)


async def check_items(client: Any, report: Report, *, images_index: str, items_index: str) -> None:
    """Every item points to an images doc that exists."""
    async for page in iter_pages(client, index=items_index, query=MATCH_ALL, source=['image_id']):
        owners: dict[str, list[str]] = {}
        for hit in page:
            report.items += 1
            owners.setdefault(str((hit.get('_source') or {}).get('image_id') or ''), []).append(
                hit['_id']
            )
        for image_id in await absent_images(client, images_index, list(owners)):
            report.items_without_image += len(owners[image_id])
            report.note('items_without_image', f'{owners[image_id][0]} -> {image_id!r}')


async def check_contents(
    client: Any, report: Report, rows: AsyncIterator[list[str]], *, images_index: str
) -> None:
    """Every exact-content row names an images doc that exists."""
    async for image_ids in rows:
        report.content_rows += len(image_ids)
        for image_id in await absent_images(client, images_index, image_ids):
            report.contents_without_image += 1
            report.note('contents_without_image', image_id)


async def audit(
    client: Any, rows: AsyncIterator[list[str]], *, images_index: str, items_index: str
) -> Report:
    report = Report()
    await check_images(client, report, images_index=images_index, items_index=items_index)
    await check_items(client, report, images_index=images_index, items_index=items_index)
    await check_contents(client, report, rows, images_index=images_index)
    return report


async def content_rows() -> AsyncIterator[list[str]]:
    """The bound project's content rows, a page of image ids at a time (read-only)."""
    from src.db.repos import image_contents
    from src.db.session import async_session_factory, project_tx
    from src.services.config_store.bound_project import bound_project_id

    project = await bound_project_id()
    after = 0
    while True:
        async with async_session_factory()() as session, project_tx(session, project):
            after, page = await image_contents.image_ids_after(session, project, after, ROWS_PAGE)
        if not page:
            return
        yield page


async def amain(args: argparse.Namespace) -> int:
    await abind_script_project(args.project)
    cfg = get_curation_config()
    client = make_script_opensearch([args.opensearch_url], timeout=120)
    try:
        report = await audit(
            client, content_rows(), images_index=cfg.images_index, items_index=cfg.items_index
        )
    finally:
        await client.close()
    print(json.dumps({'project': args.project, **asdict(report)}, indent=2))
    print(
        f'inconsistencies={report.inconsistencies} images={report.images} items={report.items} '
        f'content_rows={report.content_rows} images_without_list={report.images_without_list} '
        f'images_with_empty_list={report.images_with_empty_list}'
    )
    return 1 if report.inconsistencies else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--opensearch-url', default='http://127.0.0.1:4607')
    add_project_argument(p)
    return asyncio.run(run_with_engine(amain(p.parse_args(argv)), 'op-ingest-consistency'))


if __name__ == '__main__':
    raise SystemExit(main())
