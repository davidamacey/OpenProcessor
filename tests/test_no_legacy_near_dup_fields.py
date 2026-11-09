"""The 0.6.0 near-duplicate names are gone from ``src/``: the per-item ``dup_*`` group fields, the
unwritten ``phash``, the core index's ``duplicate_group_id`` / ``is_duplicate_primary`` /
``duplicate_score`` (now ``near_dup_*``), the per-request grouping threshold and the in-request
group assignment. Their replacements are the ``near_dup_*`` projection fields and the single-writer
jobs. (``imohash`` is the exact key again, confirmed by SHA-256 on a match, so it is not legacy.)

The 0.6.0 upgrade step names the old fields to remove them from a loaded install; the wire models
of ``/query`` keep the old response names, which are identifiers, not index fields.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / 'src'
OWNED_ELSEWHERE = ('services/upgrade/',)
LEGACY = re.compile(
    r"""(?<![A-Za-z_])(dup_group_id|dup_group_size|dup_is_representative|dup_threshold|"""
    r"""dup_method|dup_scored_at)\b|'phash'|["]phash["]"""
    r"""|['"](duplicate_group_id|is_duplicate_primary|duplicate_score)['"]"""
    r"""|DEFAULT_SIMILARITY_THRESHOLD|near_duplicate_threshold|Immich|_assign_to_duplicate_group"""
)
# The core near-duplicate paths, where no default cut is spelled as a literal and no exception
# is swallowed.
CORE_PATHS = (
    'services/duplicate_detection/',
    'services/visual_search/ingest_single.py',
    'services/visual_search/ingest_batch.py',
    'routers/ingest/',
    'routers/query_duplicates.py',
)
CUT_LITERAL = re.compile(r'(?<![\d.])0\.99(?![\d])')


def legacy_hits(src: Path) -> list[str]:
    hits = []
    for path in sorted(src.rglob('*.py')):
        rel = path.relative_to(src).as_posix()
        if rel.startswith(OWNED_ELSEWHERE):
            continue
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if LEGACY.search(line):
                hits.append(f'{rel}:{number}: {line.strip()}')
    return hits


def core_hits(src: Path) -> list[str]:
    hits = []
    for path in sorted(src.rglob('*.py')):
        rel = path.relative_to(src).as_posix()
        if not rel.startswith(CORE_PATHS):
            continue
        text = path.read_text(encoding='utf-8')
        hits += [
            f'{rel}:{n}: 0.99 as a near-duplicate cut'
            for n, line in enumerate(text.splitlines(), 1)
            if CUT_LITERAL.search(line) and 'near' in line.lower()
        ]
        if rel.startswith('services/duplicate_detection/'):
            hits += [
                f'{rel}:{node.lineno}: swallowed exception'
                for node in ast.walk(ast.parse(text))
                if isinstance(node, ast.ExceptHandler)
            ]
    return hits


def test_no_code_names_the_legacy_near_dup_fields() -> None:
    hits = legacy_hits(SRC)
    assert not hits, 'legacy near-duplicate names remain:\n' + '\n'.join(hits)


def test_the_core_paths_spell_no_cut_and_swallow_nothing() -> None:
    hits = core_hits(SRC)
    assert not hits, 'core near-duplicate path regressed:\n' + '\n'.join(hits)


def test_the_scan_finds_each_legacy_name(tmp_path: Path) -> None:
    (tmp_path / 'services' / 'duplicate_detection').mkdir(parents=True)
    planted = tmp_path / 'services' / 'duplicate_detection' / 'x.py'
    planted.write_text(
        "FIELD = 'duplicate_group_id'\n"
        'DEFAULT_SIMILARITY_THRESHOLD = 0.99  # near-duplicates, matches Immich\n'
        'try:\n    pass\nexcept Exception:\n    pass\n',
        encoding='utf-8',
    )
    assert len(legacy_hits(tmp_path)) == 2
    assert len(core_hits(tmp_path)) == 2
