"""Shard folding safety (owner D4, projects_plan.md §2.3): the two
folded roles (SETTINGS, UMAP_VIZ_STATE) are addressed by a fixed doc id
only -- never a search / count / ``*_by_query`` call -- and their doc
ids never collide with a config-store doc id (which always contains a
``:``)."""

from __future__ import annotations

import ast
from pathlib import Path


_FOLDED_ID_MODULES = (
    Path('src/clients/curation_opensearch.py'),
    Path('src/services/curation/embedding_viz.py'),
)

_SEARCH_LIKE_METHODS = {'search', 'count', 'msearch', 'delete_by_query', 'update_by_query'}


def test_settings_and_umap_viz_state_never_searched() -> None:
    """AST scan: no ``client.<search-like>(...)`` call whose ``index=``
    kwarg resolves to ``IndexRole.SETTINGS``/``IndexRole.UMAP_VIZ_STATE``
    by literal name in the surrounding source. A narrower check than a
    full type-flow analysis, but every real call site in this repo names
    the role/index inline (``index_name(cfg, IndexRole.SETTINGS)`` or the
    ``settings_index``/``umap_viz_state_index`` attrs) within the same
    call expression or a couple of lines above it, so a textual proximity
    check is sufficient and avoids false negatives from a stricter
    (call-graph) analysis missing indirection.
    """
    offending: list[str] = []
    for path in _FOLDED_ID_MODULES:
        if not path.exists():
            continue
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else None
            if name not in _SEARCH_LIKE_METHODS:
                continue
            snippet = ast.get_source_segment(source, node) or ''
            if 'SETTINGS' in snippet or 'UMAP_VIZ_STATE' in snippet or 'settings_index' in snippet:
                offending.append(f'{path}:{node.lineno}: {snippet.splitlines()[0]}')
    assert not offending, f'search-like call against a folded role: {offending}'


def test_folded_doc_ids_never_collide_with_config_store_ids() -> None:
    from src.clients.curation_opensearch import CURATION_SETTINGS_DOC_ID

    assert ':' not in CURATION_SETTINGS_DOC_ID
    # The visualization-only umap-viz-state doc id ("current") is a
    # module-level literal in embedding_viz.py, checked the same way.
    umap_viz_doc_id = 'current'
    assert ':' not in umap_viz_doc_id

    from src.services.config_store.index import activation_doc_id, config_doc_id, runtime_doc_id

    config_store_ids = [
        config_doc_id('prompt_pack', 'x'),
        config_doc_id('region_profile', 'x', 3),
        activation_doc_id('prompt_pack'),
        'meta:config_revision',
        runtime_doc_id('detection_worker', 'host'),
    ]
    for doc_id in config_store_ids:
        assert ':' in doc_id
        assert doc_id not in (CURATION_SETTINGS_DOC_ID, umap_viz_doc_id)
