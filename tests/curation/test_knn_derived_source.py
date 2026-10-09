"""Every kNN index keeps OpenSearch's derived source (vectors are not stored
again as JSON text in ``_source``, about 3x smaller on disk), and nothing
reads the nested per-box vectors through the one ``_source`` form that
returns the number ``1`` instead of each vector on such an index."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from src.clients.curation_opensearch.lifecycle import INDEX_BODIES
from src.config.region_fields import get_region_fields
from src.services.curation.region_box_embeddings import box_vector_source_includes


if TYPE_CHECKING:
    from src.config import IndexRole

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('role', list(INDEX_BODIES))
def test_every_knn_index_keeps_derived_source_on(role: IndexRole) -> None:
    index = INDEX_BODIES[role]['settings']['index']
    if not index.get('knn'):
        pytest.skip('not a kNN index')
    assert index.get('knn.derived_source.enabled', True) is True


def test_no_code_turns_derived_source_off() -> None:
    off = re.compile(r'derived_source[\w.\'"]*\s*[:=]\s*[\'"]?(False|false)')
    hits = [
        str(p.relative_to(_ROOT))
        for top in ('src', 'scripts')
        for p in (_ROOT / top).rglob('*.py')
        if off.search(p.read_text(encoding='utf-8'))
    ]
    assert hits == []


def test_box_vector_reads_name_the_leaf_paths_never_the_nested_field() -> None:
    f = get_region_fields()
    includes = box_vector_source_includes(f)
    assert f.box_embeddings not in includes
    assert f'{f.box_embeddings}.embedding' in includes


def test_no_search_includes_the_bare_nested_vector_field() -> None:
    bare = re.compile(r'(_source|source_includes)[^\n]*\bbox_embeddings\b(?![\w.])')
    hits = [
        str(p.relative_to(_ROOT))
        for top in ('src', 'scripts')
        for p in (_ROOT / top).rglob('*.py')
        if bare.search(p.read_text(encoding='utf-8'))
    ]
    assert hits == []
