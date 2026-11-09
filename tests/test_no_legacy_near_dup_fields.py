"""The 0.6.0 near-duplicate fields are gone from the curation code: the per-item `dup_*` group
fields and the unwritten `phash`. Their replacements are the `near_dup_*` projection fields.
(`imohash` is the exact key again, confirmed by SHA-256 on a match, so it is no longer legacy.)

Two owners still hold the older names and remove them in their own change: the near-duplicate
scorer under `item_scores/` (it is deleted with the `OP_CROP_DUP_THRESHOLD` setting) and the core
global index (`duplicate_group_id`; its grouping is rebuilt on the run tables). The 0.6.0 upgrade
step names the old fields to remove them from a loaded install.
"""

from __future__ import annotations

import re
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / 'src'
OWNED_ELSEWHERE = (
    'services/curation/item_scores/',
    'services/visual_search/',
    'services/duplicate_detection/',
    'services/image_admin.py',
    'clients/opensearch/',
    'routers/ingest/',
    'routers/query_duplicates.py',
    'services/upgrade/',
    'schemas/',
)
LEGACY = re.compile(
    r"""(?<![A-Za-z_])(dup_group_id|dup_group_size|dup_is_representative|dup_threshold|"""
    r"""dup_method|dup_scored_at)\b|'phash'|["]phash["]"""
)


def test_no_curation_code_names_the_legacy_near_dup_fields() -> None:
    hits = []
    for path in sorted(SRC.rglob('*.py')):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith(OWNED_ELSEWHERE):
            continue
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if LEGACY.search(line):
                hits.append(f'{rel}:{number}: {line.strip()}')
    assert not hits, 'legacy near-duplicate fields remain:\n' + '\n'.join(hits)
