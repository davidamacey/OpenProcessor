"""F-27 (fresh-start E2E findings 2026-09-25): README/CLAUDE.md doc
examples must match the real API.

Two mismatches:

1. README's Python and cURL ``POST /search/text`` examples posted a JSON
   body (``{"query": ..., "top_k": ...}``). The route
   (``src.routers.search.search_by_text``) only ever read ``text`` and
   ``top_k`` as *query* parameters -- following the README verbatim 422s
   with ``loc: ['query', 'text'], msg: 'Field required'``.
2. README's and CLAUDE.md's Face Recognition Response example showed
   ``landmarks`` as 5 ``[x, y]`` pairs. The wire model
   (``src.routers.faces.FaceBox.landmarks`` /
   ``src.schemas.visual_search.FaceDetection.landmarks``) is
   ``list[float]``: a flat 10-element list.

This introspects the real route signature and the real Pydantic field
type (source of truth) and asserts the shipped docs describe that shape,
not a hardcoded expectation that could drift the same way the docs did.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README = (REPO_ROOT / 'README.md').read_text()
CLAUDE_MD = (REPO_ROOT / 'CLAUDE.md').read_text()


def _search_text_query_param_names() -> set[str]:
    from fastapi import Query

    from src.routers.search import search_by_text

    names = set()
    for name, param in inspect.signature(search_by_text).parameters.items():
        default = param.default
        if isinstance(default, type(Query(...))):  # fastapi.params.Query
            names.add(name)
    return names


def test_search_by_text_route_takes_query_params_not_a_json_body() -> None:
    """Sanity check on the assumption the doc test below relies on."""
    names = _search_text_query_param_names()
    assert 'text' in names
    assert 'query' not in names


def test_readme_python_example_uses_query_params_for_search_text() -> None:
    section = README[README.index('Text-to-Image Search') :][:400]
    assert "params={'text'" in section, (
        "README's Text-to-Image Search example must post 'text' as a query "
        f'param (search_by_text has no request body). Got:\n{section}'
    )
    assert "json={'query'" not in section, (
        'README still shows a JSON body with a query field for /search/text, '
        'which 422s -- the route only reads query params.'
    )


def test_readme_curl_example_uses_query_params_for_search_text() -> None:
    section = README[README.index('# Text Search') :][:300]
    assert 'search/text?text=' in section, (
        f"README's cURL /search/text example must use ?text=... :\n{section}"
    )
    assert '"query"' not in section, (
        'README cURL example still posts a JSON {"query": ...} body for /search/text, which 422s.'
    )


def _face_landmarks_field_type() -> type:
    from src.routers.faces import FaceBox

    return FaceBox.model_fields['landmarks'].annotation


def test_landmarks_field_is_a_flat_list_of_floats() -> None:
    """Sanity check on the assumption the doc tests below rely on."""
    annotation = _face_landmarks_field_type()
    assert annotation == list[float], annotation


_LANDMARKS_RE = re.compile(r'"landmarks":\s*(\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*|\[[^\]]*\])')


def _assert_landmarks_example_is_flat(doc_text: str, doc_name: str) -> None:
    match = _LANDMARKS_RE.search(doc_text)
    assert match, f'{doc_name} has no landmarks example to check'
    parsed = json.loads(match.group(1))
    is_flat = bool(parsed) and isinstance(parsed[0], (int, float))
    msg = (
        f'{doc_name} shows landmarks as nested [x, y] pairs; the real wire '
        f'shape (FaceBox.landmarks: list[float]) is a flat 10-element list. '
        f'Got: {match.group(1)}'
    )
    assert is_flat, msg
    assert len(parsed) == 10, f'{doc_name} landmarks example should have 10 floats, got {parsed}'


def test_readme_face_recognition_example_landmarks_are_flat() -> None:
    _assert_landmarks_example_is_flat(README, 'README.md')


def test_claude_md_face_recognition_example_landmarks_are_flat() -> None:
    _assert_landmarks_example_is_flat(CLAUDE_MD, 'CLAUDE.md')
