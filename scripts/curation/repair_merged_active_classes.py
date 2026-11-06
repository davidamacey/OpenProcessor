#!/usr/bin/env python3
"""Re-deprecate registry classes left active after a merge.

``POST /classes/merge`` always sets ``deprecated=True`` alongside
``merged_into`` on the source class (see ``ClassRegistry.merge_class``).
But a registry hand-edited outside the API, restored from an older
snapshot, or written by a pre-fix build can end up with a class that is
*active* (``deprecated=False``) while ``merged_into`` is still set —
exactly the state ``POST /classes/{id}/restore`` now refuses to resurrect
with a ``409`` (its crops already live on the merge target; restoring the
source active would present it as a real class with no data).

This script finds every such broken entry and clears the inconsistency by
re-deprecating the source (matching what a real merge always does) — it
never touches ``merged_into``, crop data, or the target class.

Dry run by default (read-only, prints what it would change);
``--apply`` writes the registry document (through the same
``ClassRegistry.set_deprecated`` path ``POST /classes/{id}/deprecate`` uses). A
``class_registry.json`` migrated from such a file carries the state over, so
this still applies after ``openprocessor curation migrate-registry``.

    python3 scripts/curation/repair_merged_active_classes.py
    python3 scripts/curation/repair_merged_active_classes.py --apply

Talks to OpenSearch directly (``--opensearch-url``, default ``$OPENSEARCH_URL``);
the API does not need to be running.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from src.clients.curation_opensearch.registry import ClassRegistryError, get_class_registry
from src.services.projects.guard import make_script_opensearch
from src.services.projects.script_binding import add_project_argument, bind_script_project


if TYPE_CHECKING:
    from src.clients.curation_opensearch import ClassRegistryFile, RegistryClassEntry


def find_broken(registry: ClassRegistryFile) -> list[RegistryClassEntry]:
    """Active classes (``deprecated=False``) that still carry ``merged_into``."""
    return [c for c in registry.classes if not c.deprecated and c.merged_into is not None]


def format_entry(entry: RegistryClassEntry) -> str:
    return (
        f'class_id={entry.class_id} name={entry.class_name!r} '
        f'merged_into={entry.merged_into} deprecated=False -> True'
    )


async def run(client: Any, *, apply: bool, verbose: bool) -> int:
    store = get_class_registry()
    broken = find_broken(await store.snapshot(client))
    if verbose:
        for entry in broken:
            print(format_entry(entry))
    print(f'{len(broken)} active class(es) with merged_into set')
    if not broken:
        return 0
    if not apply:
        print('Dry-run only. Pass --apply to write.')
        return 0

    errors = 0
    for entry in broken:
        try:
            await store.set_deprecated(client, entry.class_id, True)
        except ClassRegistryError as exc:
            print(f'error: class_id={entry.class_id}: {exc}', file=sys.stderr)
            errors += 1
    fixed = len(broken) - errors
    print(f'fixed={fixed} errors={errors}')
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        '--opensearch-url', default=os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')
    )
    p.add_argument('--verbose', action='store_true', help='Print one line per broken class.')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--dry-run', action='store_true', default=True)
    g.add_argument('--apply', dest='dry_run', action='store_false')
    add_project_argument(p)
    return p


async def _amain(args: argparse.Namespace) -> int:
    client = make_script_opensearch([args.opensearch_url], use_ssl=False, timeout=60)
    try:
        return await run(client, apply=not args.dry_run, verbose=args.verbose)
    finally:
        await client.close()


def main() -> int:
    args = build_parser().parse_args()
    bind_script_project(args.project)
    return asyncio.run(_amain(args))


if __name__ == '__main__':
    sys.exit(main())
