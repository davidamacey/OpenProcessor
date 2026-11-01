#!/usr/bin/env python3
"""Report or remove every stored face record (#379).

Deletes the ``visual_search_faces`` index (face embeddings, person ids and names), the face
FAISS cluster files, and the face link on ``visual_search_people`` documents. The people
documents themselves stay. It runs through ``./openprocessor faces purge``; there is no API
route, because the API has no authentication.

    python -m scripts.purge_face_data --dry-run   # counts, changes nothing
    python -m scripts.purge_face_data             # counts, then refuses (nothing deleted)
    python -m scripts.purge_face_data --yes       # delete; requires OP_FACES_ENABLED=false

Exit codes: 0 done (or dry run), 1 the purge failed or left data behind, 2 refused.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402

from opensearchpy import OpenSearchException

from src.config import get_settings
from src.config.cluster_index import FAISS_INDEX_DIR
from src.config.faces import faces_enabled
from src.core.logging import get_logger
from src.services.face_privacy import (
    FaceDataCounts,
    FacePurgeIncompleteError,
    face_data_counts,
    purge_face_data,
)
from src.services.projects.guard import make_script_opensearch


logger = get_logger(__name__)

AUDIT_FILE = Path('data/audit/faces_purge.jsonl')


def make_client() -> Any:
    """The guarded script client. Unbound, so it can reach only indexes no project owns,
    which is exactly the deployment-wide face store."""
    return make_script_opensearch([get_settings().opensearch_url])


def _audit(before: FaceDataCounts, index_deleted: bool) -> None:
    """One counts-only record, to the log and to ``data/audit/faces_purge.jsonl``. Never an id,
    a name, a path or a vector."""
    counts = {
        'face_docs': before.face_docs,
        'persons': before.persons,
        'named_persons': before.named_persons,
        'thumbnails': before.thumbnails,
        'people_face_links': before.people_face_links,
        'faiss_files': len(before.faiss_files),
        'index_deleted': index_deleted,
    }
    logger.warning('faces_purged', **counts)
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {**counts, 'at': datetime.now(UTC).isoformat(), 'version': get_settings().api_version}
    with AUDIT_FILE.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(record) + '\n')


async def _run(args: argparse.Namespace) -> int:
    client = make_client()
    try:
        if args.dry_run or not args.yes:
            counts = await face_data_counts(client, args.index_dir)
            if args.dry_run:
                logger.info('faces_purge_dry_run', face_docs=counts.face_docs)
            print(json.dumps(asdict(counts)))
            if args.dry_run:
                return 0
            print('refusing: re-run with --yes to delete this', file=sys.stderr)
            return 2
        if faces_enabled():
            print(
                'refusing: set OP_FACES_ENABLED=false in .env and restart the api first, so '
                'nothing writes new face data while it is deleted',
                file=sys.stderr,
            )
            return 2
        try:
            before, after = await purge_face_data(client, args.index_dir)
        except FacePurgeIncompleteError as exc:
            print(json.dumps({'after': asdict(exc.after)}), file=sys.stderr)
            return 1
        except OpenSearchException as exc:
            print(f'purge failed: {type(exc).__name__}', file=sys.stderr)
            return 1
        _audit(before, index_deleted=before.face_index_exists)
        print(json.dumps({'before': asdict(before), 'after': asdict(after)}))
        return 0
    finally:
        await client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='print the counts; change nothing')
    mode.add_argument('--yes', action='store_true', help='delete (requires OP_FACES_ENABLED=false)')
    parser.add_argument('--index-dir', type=Path, default=FAISS_INDEX_DIR)
    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == '__main__':
    sys.exit(main())
