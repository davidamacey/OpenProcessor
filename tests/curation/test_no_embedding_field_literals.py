"""No reader names a vector field: it asks the registry (embeddings plan 5.4).

``ITEM_EMBEDDING_FIELD`` and ``DEFAULT_KNN_FIELD`` are gone, and a ``pe_embedding`` or
``region_box_embeddings`` string literal outside the registry modules, the 0.6.0 upgrade loader
and the mapping builders is a reader that would keep reading the old space after a flip.
Docstrings and comments are not code; the scan walks the syntax tree.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

_LITERAL = re.compile(r'\b(pe_embedding|region_box_embeddings)\b')
_RETIRED_NAMES = frozenset({'ITEM_EMBEDDING_FIELD', 'DEFAULT_KNN_FIELD'})

# path prefix -> why that module may name a physical field.
ALLOWED: dict[str, str] = {
    'src/domain/embedding_spaces.py': 'the registry vocabulary: the 0.6.0 default names',
    'src/services/embedding_spaces/': 'the registry service and the re-index writer',
    'src/services/upgrade/': 'the one-step 0.6.0 upgrade reads the old layout',
    'src/db/': 'registry rows and migrations',
    'src/clients/curation_opensearch/bodies_core.py': 'mapping builder: index creation',
    'src/clients/curation_opensearch/ensure_fields.py': 'mapping builder: additive field repair',
    'src/config/region_fields.py': 'the storage name of the nested box vectors (RegionFields)',
    'scripts/db/': 'builders of the 0.6.0 fixture indexes',
    'scripts/codegen/': 'the guards themselves',
}


# Readers that still name a vector field. A ratchet like the baselines of scripts/codegen: the
# list only shrinks (``test_pending_entries_still_offend`` fails for a file that was converted),
# and a new offender fails the first test.
PENDING: dict[str, str] = {
    'src/clients/occ.py': 'OCC_BULK_MGET_SOURCE_EXCLUDES: every OCC caller passes the vector excludes',
    'src/config/__init__.py': 'exports ITEM_EMBEDDING_FIELD until the last reader is converted',
    'src/config/curation.py': 'defines ITEM_EMBEDDING_FIELD',
    'src/domain/embedding_state.py': 'the clause helpers default to the constant',
    'src/services/curation/wire.py': 'item_source_excludes: every serializer caller passes the excludes',
    'src/services/curation/review_tab_clauses.py': 'the all tab: build_tab_query takes the field',
    'src/services/curation/strategy_builders.py': 'the requires_field descriptor of semantic search',
    'scripts/bench/export_project_embeddings.py': 'standalone httpx harness over a 0.6.0 export',
    'scripts/bench/replicate_items.py': 'standalone httpx harness over a 0.6.0 export',
}


def _allowed(rel: str) -> bool:
    return any(rel.startswith(prefix) for prefix in ALLOWED)


def _scan() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for root in ('src', 'scripts'):
        for path in sorted((REPO_ROOT / root).rglob('*.py')):
            rel = str(path.relative_to(REPO_ROOT))
            if not _allowed(rel) and (hits := _offences(path)):
                found[rel] = hits
    return found


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            first = body[0] if body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                ids.add(id(first.value))
    return ids


def _offences(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    docstrings = _docstring_nodes(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings and _LITERAL.search(node.value):
                found.append(f'{path.relative_to(REPO_ROOT)}:{node.lineno} literal {node.value!r}')
        elif isinstance(node, ast.Name) and node.id in _RETIRED_NAMES:
            found.append(f'{path.relative_to(REPO_ROOT)}:{node.lineno} {node.id}')
        elif isinstance(node, ast.Attribute) and node.attr in _RETIRED_NAMES:
            found.append(f'{path.relative_to(REPO_ROOT)}:{node.lineno} {node.attr}')
        elif isinstance(node, ast.alias) and node.name in _RETIRED_NAMES:
            found.append(f'{path.relative_to(REPO_ROOT)}:{node.lineno} import {node.name}')
    return found


def test_no_new_reader_names_a_vector_field() -> None:
    offenders = [h for rel, hits in _scan().items() if rel not in PENDING for h in hits]
    assert offenders == [], (
        'readers must resolve their field with embedding_space():\n' + '\n'.join(offenders)
    )


def test_pending_entries_still_offend() -> None:
    converted = sorted(set(PENDING) - set(_scan()))
    assert converted == [], f'converted: delete from PENDING: {converted}'


def test_allowed_prefixes_exist() -> None:
    stale = [p for p in ALLOWED if not (REPO_ROOT / p).exists()]
    assert stale == []
