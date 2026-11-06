"""The 0.6.0 near-duplicate fields are gone from the curation code: the per-item `dup_*` group
fields, the unwritten `phash` and the sampled `imohash` key. Their replacements are the
`near_dup_*` projection fields and `content_sha256`.

Three owners still hold the older names and remove them in their own change: the near-duplicate
scorer under `item_scores/` (it is deleted with the `OP_CROP_DUP_THRESHOLD` setting), the core
global index (`duplicate_group_id` and the core `imohash`; its grouping is rebuilt on the run
tables) and the project combine with its source reader, which dedupe on the stored `imohash` of
the source images.
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
    'schemas/',
    'services/projects/combine/',
    'services/curation/dataset_import/project_source.py',
)
LEGACY = re.compile(
    r"""(?<![A-Za-z_])(dup_group_id|dup_group_size|dup_is_representative|dup_threshold|"""
    r"""dup_method|dup_scored_at)\b|'phash'|"phash"|\bimohash\b"""
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
