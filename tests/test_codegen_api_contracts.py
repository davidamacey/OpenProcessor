"""Generated API contracts under contracts/: in sync, env-independent, drift-checked."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel, Field

from src.services.curation import wire
from src.services.curation.class_sources import CLASS_SOURCE_ROLES


if TYPE_CHECKING:
    from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
CODEGEN = REPO_ROOT / 'scripts' / 'codegen'
CONTRACTS = REPO_ROOT / 'contracts'


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f'_test_{name}', CODEGEN / f'{name}.py')
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def mod() -> ModuleType:
    return _load('export_api_contracts')


@pytest.fixture(scope='module')
def rendered(mod: ModuleType) -> dict[Path, str]:
    return mod.render()


def test_umbrella_check_passes_on_committed_output_despite_env_overrides() -> None:
    # Deployment overrides in the caller's shell must not leak into the
    # output: the generator re-runs itself with a scrubbed env.
    env = {
        **os.environ,
        'OP_API_PREFIX': '/elsewhere',
        'OP_API_TAG': 'Other',
        'OP_REGION_FIELD_STATUS': 'legacy_status',
    }
    proc = subprocess.run(
        [sys.executable, str(CODEGEN / 'generate_contracts.py'), '--check'],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_write_then_check_roundtrip(
    mod: ModuleType, rendered: dict[Path, str], tmp_path: Path
) -> None:
    mod.write(rendered, tmp_path)
    assert mod.check(rendered, tmp_path) == []
    assert mod.run(tmp_path, check_only=True, only=['class-sources']) == 0


@pytest.mark.parametrize(
    'rel', ['json/item_wire.json', 'ts/itemWire.ts', 'ts/classSources.ts', 'openapi/curation.json']
)
def test_check_fails_on_stale_file(
    mod: ModuleType, rendered: dict[Path, str], tmp_path: Path, rel: str
) -> None:
    mod.write(rendered, tmp_path)
    target = tmp_path / rel
    target.write_text(target.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    problems = mod.check(rendered, tmp_path)
    assert problems == [f'{target} is out of sync']


def test_check_fails_on_missing_file(
    mod: ModuleType, rendered: dict[Path, str], tmp_path: Path
) -> None:
    problems = mod.check(rendered, tmp_path)
    assert len(problems) == len(rendered)
    assert all('does not exist' in p for p in problems)


def test_check_detects_new_wire_key(
    mod: ModuleType, rendered: dict[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key added to the item contract must make the committed files fail."""
    mod.write(rendered, tmp_path)
    real = mod._item_wire_facts

    def facts_with_new_key() -> dict[str, Any]:
        facts = real()
        facts['item_keys'] = [*facts['item_keys'], 'new_key']
        facts['json_schema']['properties']['new_key'] = {'type': 'string'}
        return facts

    monkeypatch.setattr(mod, '_item_wire_facts', facts_with_new_key)
    assert mod.run(tmp_path, check_only=True, only=['item-wire']) == 1


def test_itemdoc_serializer_divergence_is_fatal(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wire, 'ITEM_WIRE_KEYS', wire.ITEM_WIRE_KEYS | {'emitted_but_undocumented'})
    with pytest.raises(SystemExit, match='emitted_but_undocumented'):
        mod._item_wire_facts()


def test_committed_item_wire_matches_serializer() -> None:
    data = json.loads((CONTRACTS / 'json' / 'item_wire.json').read_text(encoding='utf-8'))
    assert set(data['item_keys']) == set(wire.ITEM_WIRE_KEYS)
    assert len(data['item_keys']) == len(wire.ITEM_WIRE_KEYS)
    assert data['region_keys'] == list(wire.REGION_WIRE_KEYS)
    assert all(k.startswith('region_') for k in data['region_keys'])
    assert set(data['extra_keys']['review']) == wire.REVIEW_EXTRA_KEYS
    ts = (CONTRACTS / 'ts' / 'itemWire.ts').read_text(encoding='utf-8')
    for key in data['item_keys']:
        assert f'  {key}: ' in ts
        assert f"  '{key}'," in ts


def test_committed_class_source_roles() -> None:
    ts = (CONTRACTS / 'ts' / 'classSources.ts').read_text(encoding='utf-8')
    block = ts.split('export const CLASS_SOURCE_ROLES = [')[1].split('] as const')[0]
    assert [line.strip().strip(",'") for line in block.strip().splitlines()] == list(
        CLASS_SOURCE_ROLES
    )


def test_committed_openapi_is_a_closed_curation_subset() -> None:
    spec = json.loads((CONTRACTS / 'openapi' / 'curation.json').read_text(encoding='utf-8'))
    prefix = spec['x-api-prefix']
    assert prefix == '/curation'
    assert spec['paths']
    assert all(p == prefix or p.startswith(prefix + '/') for p in spec['paths'])
    refs: set[str] = set()
    _refs(spec, refs)
    defined = {f'#/components/{s}/{n}' for s, names in spec['components'].items() for n in names}
    assert refs == defined, 'every $ref resolves and every component is referenced'
    assert 'ItemDoc' in spec['components']['schemas']


def _refs(node: Any, out: set[str]) -> None:
    if isinstance(node, dict):
        if isinstance(node.get('$ref'), str):
            out.add(node['$ref'])
        for v in node.values():
            _refs(v, out)
    elif isinstance(node, list):
        for v in node:
            _refs(v, out)


def test_filter_openapi_prunes_paths_and_components(mod: ModuleType) -> None:
    def ref(name: str) -> dict[str, str]:
        return {'$ref': f'#/components/schemas/{name}'}

    spec = {
        'openapi': '3.1.0',
        'info': {'title': 'App', 'version': '9.9.9'},
        'paths': {
            '/curation/a': {
                'get': {'responses': {'200': {'content': {'x': {'schema': ref('A')}}}}}
            },
            '/curationx/b': {
                'get': {'responses': {'200': {'content': {'x': {'schema': ref('B')}}}}}
            },
            '/v1/curation/a': {
                'get': {'responses': {'200': {'content': {'x': {'schema': ref('B')}}}}}
            },
            '/detect': {'get': {'responses': {'200': {'content': {'x': {'schema': ref('B')}}}}}},
        },
        'components': {
            'schemas': {
                'A': {'properties': {'c': ref('C')}},
                'B': {'type': 'object'},
                'C': {'items': ref('A')},
            }
        },
    }
    out = mod.filter_openapi(spec, '/curation')
    assert list(out['paths']) == ['/curation/a']
    assert sorted(out['components']['schemas']) == ['A', 'C']
    assert '9.9.9' not in json.dumps(out)


def test_filter_openapi_rejects_external_refs(mod: ModuleType) -> None:
    spec = {
        'openapi': '3.1.0',
        'info': {'title': 'App', 'version': '1'},
        'paths': {'/curation/a': {'get': {'x': {'$ref': 'other.json#/Thing'}}}},
        'components': {},
    }
    with pytest.raises(ValueError, match='unsupported \\$ref'):
        mod.filter_openapi(spec, '/curation')


@pytest.mark.parametrize(
    ('schema', 'expected'),
    [
        ({'type': 'string'}, 'string'),
        ({'type': 'integer'}, 'number'),
        ({'anyOf': [{'type': 'integer'}, {'type': 'null'}]}, 'number | null'),
        ({'type': 'array', 'items': {'type': 'number'}}, 'number[]'),
        (
            {'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'null'}]}},
            '(string | null)[]',
        ),
        (
            {'type': 'array', 'prefixItems': [{'type': 'number'}, {'type': 'string'}]},
            '[number, string]',
        ),
        ({}, 'unknown'),
        ({'type': 'object'}, 'Record<string, unknown>'),
    ],
)
def test_ts_type(mod: ModuleType, schema: dict[str, Any], expected: str) -> None:
    assert mod.ts_type(schema) == expected


def test_ts_type_rejects_refs(mod: ModuleType) -> None:
    with pytest.raises(ValueError, match='unsupported'):
        mod.ts_type({'$ref': '#/$defs/Thing'})


def test_python_override_env_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _load('_contract_runtime')
    monkeypatch.setenv(runtime.PYTHON_OVERRIDE_ENV, '/opt/custom/python')
    assert runtime.resolve_python() == '/opt/custom/python'


class _Leaf(BaseModel):
    text: str | None = None
    box: list[float] | None = None


class _Middle(BaseModel):
    leaves: list[_Leaf] = Field(default_factory=list)
    best: _Leaf | None = None


class _Node(BaseModel):
    name: str
    children: list[_Node] = Field(default_factory=list)


class _Root(BaseModel):
    id: str
    middles: list[_Middle] = Field(default_factory=list)
    maybe_middles: list[_Middle] | None = None
    tree: _Node | None = None


def test_ref_types_render_by_name(mod: ModuleType) -> None:
    refs: set[str] = set()
    assert mod.ts_type({'$ref': '#/$defs/Thing'}, refs) == 'Thing'
    assert mod.ts_type({'type': 'array', 'items': {'$ref': '#/$defs/A'}}, refs) == 'A[]'
    assert (
        mod.ts_type(
            {'anyOf': [{'type': 'array', 'items': {'$ref': '#/$defs/B'}}, {'type': 'null'}]}, refs
        )
        == 'B[] | null'
    )
    assert refs == {'Thing', 'A', 'B'}


def test_nested_refs_emit_sorted_interfaces_transitively(mod: ModuleType) -> None:
    schema = _Root.model_json_schema(mode='serialization')
    refs: set[str] = set()
    fields = {k: mod.ts_type(v, refs) for k, v in schema['properties'].items()}
    assert fields == {
        'id': 'string',
        'middles': '_Middle[]',
        'maybe_middles': '_Middle[] | null',
        'tree': '_Node | null',
    }
    text = '\n'.join(mod.ts_interfaces(schema, refs))
    # _Leaf is only reachable through _Middle; _Node refers to itself.
    assert text == '\n'.join(
        [
            'export interface _Leaf {',
            '  text: string | null;',
            '  box: number[] | null;',
            '}',
            '',
            'export interface _Middle {',
            '  leaves: _Leaf[];',
            '  best: _Leaf | null;',
            '}',
            '',
            'export interface _Node {',
            '  name: string;',
            '  children: _Node[];',
            '}',
            '',
        ]
    )


def test_ref_to_missing_def_is_fatal(mod: ModuleType) -> None:
    with pytest.raises(ValueError, match='undefined'):
        mod.ts_interfaces({'$defs': {}}, {'Ghost'})


def test_committed_item_wire_ts_declares_nested_models() -> None:
    ts = (CONTRACTS / 'ts' / 'itemWire.ts').read_text(encoding='utf-8')
    schema = json.loads((CONTRACTS / 'json' / 'item_wire.json').read_text(encoding='utf-8'))[
        'json_schema'
    ]
    for name in schema.get('$defs', {}):
        assert f'export interface {name} {{' in ts
