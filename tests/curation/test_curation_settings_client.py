"""Tests for the shared curation-settings document client
(``src.clients.curation_opensearch.get_curation_settings`` /
``update_curation_settings`` — curation deployment-settings plan).

A tiny in-memory fake stands in for OpenSearch's ``get``/``update``
behavior: ``get`` raises a duck-typed "not found" error for a missing
doc id (mirrors ``opensearchpy.NotFoundError``'s message shape, per
``image_serving.fetch_crop_source``'s own convention), and ``update``
reproduces OpenSearch's real partial-update semantics -- the ``doc``
merges recursively into an existing *object* field (``defaults``) rather
than replacing it outright, which is exactly the behavior a partial
``PUT /curation/settings`` depends on to avoid clobbering other axes.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.clients.curation_opensearch import (
    CURATION_SETTINGS_DOC_ID,
    get_curation_settings,
    update_curation_settings,
)
from src.config.curation import CurationConfig


class _NotFoundError(Exception):
    pass


class FakeSettingsOpenSearch:
    """Doc-store fake covering exactly the two calls the settings client
    makes -- ``get`` and ``update`` -- with OpenSearch's real partial-merge
    semantics for object fields."""

    def __init__(self) -> None:
        self._docs: dict[tuple[str, str], dict[str, Any]] = {}

    async def get(self, index: str, id: str) -> dict[str, Any]:  # noqa: A002 - mirrors opensearchpy's kwarg name
        key = (index, id)
        if key not in self._docs:
            raise _NotFoundError(f'[404] not found: {index}/{id}')
        return {'_source': dict(self._docs[key])}

    async def update(
        self,
        index: str,
        id: str,  # noqa: A002 - mirrors opensearchpy's kwarg name
        body: dict[str, Any],
        refresh: bool = False,  # noqa: ARG002 - accepted to mirror the real client's signature
    ) -> dict[str, Any]:
        key = (index, id)
        existing = dict(self._docs.get(key, {}))
        incoming = body['doc']
        for field, value in incoming.items():
            if field == 'defaults' and isinstance(value, dict):
                merged_defaults = dict(existing.get('defaults') or {})
                merged_defaults.update(value)
                existing['defaults'] = merged_defaults
            else:
                existing[field] = value
        self._docs[key] = existing
        return {'result': 'updated' if key in self._docs else 'created'}


@pytest.fixture
def fake_os() -> FakeSettingsOpenSearch:
    return FakeSettingsOpenSearch()


@pytest.fixture
def cfg() -> CurationConfig:
    return CurationConfig()


@pytest.mark.asyncio
async def test_get_with_no_doc_returns_empty_defaults(
    fake_os: FakeSettingsOpenSearch, cfg: CurationConfig
) -> None:
    doc = await get_curation_settings(fake_os, cfg=cfg)
    assert doc == {'defaults': {}, 'updated_at': None, 'updated_by': None}


@pytest.mark.asyncio
async def test_put_creates_the_document(
    fake_os: FakeSettingsOpenSearch, cfg: CurationConfig
) -> None:
    doc = await update_curation_settings(fake_os, {'cluster': 'ahc'}, cfg=cfg)
    assert doc['defaults'] == {'cluster': 'ahc'}
    assert doc['updated_at'] is not None
    assert doc['updated_by'] is None


@pytest.mark.asyncio
async def test_put_again_partially_updates_without_clobbering_other_axes(
    fake_os: FakeSettingsOpenSearch, cfg: CurationConfig
) -> None:
    await update_curation_settings(fake_os, {'cluster': 'ahc', 'sort': 'atypicality'}, cfg=cfg)
    doc = await update_curation_settings(fake_os, {'cluster': 'ivf'}, cfg=cfg)
    assert doc['defaults'] == {'cluster': 'ivf', 'sort': 'atypicality'}


@pytest.mark.asyncio
async def test_put_with_empty_defaults_is_a_no_op_merge(
    fake_os: FakeSettingsOpenSearch, cfg: CurationConfig
) -> None:
    """An empty ``defaults`` dict has no keys to merge -- existing axes must
    survive untouched (this is what lets the router's future no-op PUTs be
    harmless)."""
    await update_curation_settings(fake_os, {'cluster': 'ahc'}, cfg=cfg)
    doc = await update_curation_settings(fake_os, {}, cfg=cfg)
    assert doc['defaults'] == {'cluster': 'ahc'}


@pytest.mark.asyncio
async def test_get_reflects_a_prior_update(
    fake_os: FakeSettingsOpenSearch, cfg: CurationConfig
) -> None:
    await update_curation_settings(fake_os, {'prompt_pack': 'generic_item_v1'}, cfg=cfg)
    doc = await get_curation_settings(fake_os, cfg=cfg)
    assert doc['defaults'] == {'prompt_pack': 'generic_item_v1'}


@pytest.mark.asyncio
async def test_put_null_clears_an_axis_without_deleting_the_document(
    fake_os: FakeSettingsOpenSearch, cfg: CurationConfig
) -> None:
    await update_curation_settings(fake_os, {'cluster': 'ahc', 'sort': 'atypicality'}, cfg=cfg)
    doc = await update_curation_settings(fake_os, {'sort': None}, cfg=cfg)
    assert doc['defaults'] == {'cluster': 'ahc'}


@pytest.mark.asyncio
async def test_uses_the_fixed_singleton_doc_id(
    fake_os: FakeSettingsOpenSearch, cfg: CurationConfig
) -> None:
    from src.config import IndexRole, index_name

    await update_curation_settings(fake_os, {'cluster': 'hdbscan'}, cfg=cfg)
    index = index_name(cfg, IndexRole.SETTINGS)
    assert (index, CURATION_SETTINGS_DOC_ID) in fake_os._docs


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
