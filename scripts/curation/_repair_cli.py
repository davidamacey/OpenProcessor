"""The command line the item repair scripts share: the index/paging/dry-run options, the
OpenSearch client lifecycle and the exit code (1 when any write errored)."""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.db.engine import run_with_engine
from src.services.projects.guard import make_script_opensearch
from src.services.projects.script_binding import add_project_argument, bind_script_project


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')


def build_repair_parser(
    description: str | None,
    add_options: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description,
        epilog=(
            'Examples: %(prog)s | %(prog)s --verbose | %(prog)s --apply. '
            'The index is CurationConfig.items_index unless --index is given.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--index', default=get_curation_config().items_index)
    p.add_argument('--crop-id-prefix', default=None, help='Only crop ids starting with this.')
    p.add_argument('--page-size', type=int, default=500)
    p.add_argument('--opensearch-url', default=DEFAULT_OPENSEARCH)
    p.add_argument('--verbose', action='store_true', help='Print one line per candidate.')
    if add_options is not None:
        add_options(p)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--dry-run', action='store_true', default=True)
    g.add_argument('--apply', dest='dry_run', action='store_false')
    add_project_argument(p)
    return p


def repair_main(
    build_parser: Callable[[], argparse.ArgumentParser],
    run: Callable[[argparse.Namespace, Any], Awaitable[int]],
    engine_name: str,
) -> int:
    args = build_parser().parse_args()
    bind_script_project(args.project)
    if args.page_size <= 0:
        build_parser().error('--page-size must be positive')

    async def _async_main() -> int:
        client = make_script_opensearch([args.opensearch_url], use_ssl=False, timeout=300)
        try:
            return await run(args, client)
        finally:
            await client.close()

    return asyncio.run(run_with_engine(_async_main(), engine_name))


def exit_code(written: dict[str, int]) -> int:
    print(' '.join(f'{k}={v}' for k, v in written.items()))
    return 1 if written['errors'] else 0
