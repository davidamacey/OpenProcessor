"""Every structured error carries a human ``message``.

Two walks: the committed OpenAPI contract (every declared 4xx/5xx body whose
``detail`` is an object has ``message``), and the router source (every
``HTTPException(detail={...})`` literal has a ``message`` key), because a raise
site need not be declared as a response."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'contracts' / 'openapi' / 'curation.json'


def _resolve(schema: dict[str, Any], comps: dict[str, Any]) -> dict[str, Any]:
    while '$ref' in schema:
        schema = comps[schema['$ref'].rsplit('/', 1)[-1]]
    return schema


def _variants(schema: dict[str, Any], comps: dict[str, Any]) -> list[dict[str, Any]]:
    schema = _resolve(schema, comps)
    for key in ('anyOf', 'oneOf'):
        if key in schema:
            return [v for s in schema[key] for v in _variants(s, comps)]
    return [schema]


def test_declared_error_bodies_with_an_object_detail_have_a_message() -> None:
    doc = json.loads(CONTRACT.read_text())
    comps = doc['components']['schemas']
    missing: list[str] = []
    for path, ops in doc['paths'].items():
        for method, op in ops.items():
            for code, resp in (op.get('responses') or {}).items():
                schema = ((resp.get('content') or {}).get('application/json') or {}).get('schema')
                if not code.startswith(('4', '5')) or schema is None:
                    continue
                detail = (_resolve(schema, comps).get('properties') or {}).get('detail')
                if detail is None:
                    continue
                for variant in _variants(detail, comps):
                    props = variant.get('properties')
                    if props is not None and 'message' not in props:
                        missing.append(f'{method.upper()} {path} {code}')
    assert not missing, missing


@pytest.mark.parametrize('base', ['src/routers', 'src/services'])
def test_every_dict_detail_literal_has_a_message(base: str) -> None:
    missing: list[str] = []
    for file in sorted((ROOT / base).rglob('*.py')):
        for node in ast.walk(ast.parse(file.read_text())):
            if not (
                isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'HTTPException'
            ):
                continue
            detail = next((k.value for k in node.keywords if k.arg == 'detail'), None)
            if isinstance(detail, ast.Dict):
                keys = {k.value for k in detail.keys if isinstance(k, ast.Constant)}
                if 'message' not in keys:
                    missing.append(f'{file.relative_to(ROOT)}:{node.lineno}')
    assert not missing, missing
