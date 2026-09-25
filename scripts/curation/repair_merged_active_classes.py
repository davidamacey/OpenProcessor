#!/usr/bin/env python3
"""Re-deprecate registry classes left active after a merge (F-56).

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
``--apply`` writes the registry (through the same
``ClassRegistry.set_deprecated`` path ``POST /classes/{id}/deprecate``
uses, so a snapshot of the prior file is taken first).

    python3 scripts/curation/repair_merged_active_classes.py
    python3 scripts/curation/repair_merged_active_classes.py --apply

The registry path comes from ``CurationConfig.class_registry_path``
unless ``--registry`` is given. No OpenSearch access — this only touches
``class_registry.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from src.clients.curation_opensearch import ClassRegistry, ClassRegistryError, RegistryClassEntry
from src.config import get_curation_config


def find_broken(registry: ClassRegistry) -> list[RegistryClassEntry]:
    """Active classes (``deprecated=False``) that still carry ``merged_into``."""
    return [c for c in registry.load().classes if not c.deprecated and c.merged_into is not None]


def format_entry(entry: RegistryClassEntry) -> str:
    return (
        f'class_id={entry.class_id} name={entry.class_name!r} '
        f'merged_into={entry.merged_into} deprecated=False -> True'
    )


def run(registry: ClassRegistry, *, apply: bool, verbose: bool) -> int:
    broken = find_broken(registry)
    if verbose:
        for entry in broken:
            print(format_entry(entry))
    print(f'{len(broken)} active class(es) with merged_into set (registry: {registry.path})')
    if not broken:
        return 0
    if not apply:
        print('Dry-run only. Pass --apply to write.')
        return 0

    errors = 0
    for entry in broken:
        try:
            registry.set_deprecated(entry.class_id, True)
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
        '--registry',
        default=None,
        help='class_registry.json path (default: CurationConfig.class_registry_path).',
    )
    p.add_argument('--verbose', action='store_true', help='Print one line per broken class.')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--dry-run', action='store_true', default=True)
    g.add_argument('--apply', dest='dry_run', action='store_false')
    return p


def main() -> int:
    args = build_parser().parse_args()
    registry_path = args.registry or get_curation_config().class_registry_path
    registry = ClassRegistry(registry_path)
    return run(registry, apply=not args.dry_run, verbose=args.verbose)


if __name__ == '__main__':
    sys.exit(main())
