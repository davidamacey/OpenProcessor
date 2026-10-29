"""List the curation projects of a stack with item, validated and holdout counts (read only).

    setp_inventory.py --api http://127.0.0.1:4603 --out artifacts_local/cluster_search/w0/setp_inventory.json

The owner decides which projects are set P; this only reports what exists. The output holds
project slugs and counts, so it stays under the gitignored ``artifacts_local/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx


if TYPE_CHECKING:
    from collections.abc import Sequence

COUNT_FIELDS = ('items', 'images', 'validated', 'holdout_items', 'classes')


def collect(
    api: str, prefix: str, transport: httpx.BaseTransport | None = None
) -> list[dict[str, Any]]:
    rows = []
    with httpx.Client(timeout=httpx.Timeout(120.0), transport=transport) as client:
        listing = client.get(f'{api}{prefix}/projects', params={'include_archived': 'true'})
        listing.raise_for_status()
        for project in listing.json()['projects']:
            stats = client.get(f'{api}{prefix}/projects/{project["slug"]}/stats')
            stats.raise_for_status()
            counts = stats.json()['counts']
            rows.append(
                {
                    'slug': project['slug'],
                    'status': project['status'],
                    **{f: counts.get(f) for f in COUNT_FIELDS},
                }
            )
    return rows


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'projects': rows}, indent=2) + '\n')


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--api', required=True)
    p.add_argument('--prefix', default='/curation')
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    rows = collect(args.api.rstrip('/'), args.prefix.rstrip('/'))
    write(Path(args.out), rows)
    print(json.dumps({'projects': len(rows), 'out': args.out}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
