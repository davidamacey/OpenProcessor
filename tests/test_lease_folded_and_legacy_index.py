"""The ``lease`` and ``class_registry`` object keys are mapped on the configs index and on a
legacy settings index, and the bootstrap migration puts them exactly once per index."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.clients.curation_opensearch import ensure_overlay_fields
from src.clients.curation_opensearch.bodies_other import _configs_body, _settings_body
from src.clients.curation_opensearch.ensure_overlay_fields import ensure_settings_object_keys
from src.services.curation import index_bootstrap


_KEYS = ('lease', 'class_registry')


class _Indices:
    def __init__(self) -> None:
        self.puts: list[tuple[str, dict[str, Any]]] = []

    async def put_mapping(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.puts.append((index, body))
        return {'acknowledged': True}


class _Client:
    def __init__(self) -> None:
        self.indices = _Indices()


@pytest.mark.parametrize('body', [_configs_body, _settings_body])
def test_mapping_has_object_keys(body: Any) -> None:
    props = body()['mappings']['properties']
    for key in _KEYS:
        assert props[key] == {'type': 'object', 'enabled': False}


def test_migration_is_in_the_bootstrap_list() -> None:
    assert 'settings_object_keys' in [name for name, _ in index_bootstrap._migrations()]


@pytest.mark.asyncio
async def test_folded_project_puts_the_keys_once(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SimpleNamespace(configs_index='op_prj_a__configs', settings_index='op_prj_a__configs')
    monkeypatch.setattr(ensure_overlay_fields, 'config', cfg)
    client = _Client()

    out = await ensure_settings_object_keys(client)  # type: ignore[arg-type]

    assert out['acknowledged'] is True
    assert [i for i, _ in client.indices.puts] == ['op_prj_a__configs']
    assert set(client.indices.puts[0][1]['properties']) == set(_KEYS)


@pytest.mark.asyncio
async def test_legacy_project_puts_the_keys_on_both_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SimpleNamespace(configs_index='op_prj_a__configs', settings_index='op_prj_a__settings')
    monkeypatch.setattr(ensure_overlay_fields, 'config', cfg)
    client = _Client()

    await ensure_settings_object_keys(client)  # type: ignore[arg-type]

    assert [i for i, _ in client.indices.puts] == ['op_prj_a__configs', 'op_prj_a__settings']
