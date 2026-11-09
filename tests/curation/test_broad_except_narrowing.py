"""Handlers narrowed to the library's exception type: a transport failure keeps its
documented fallback, any other error is a bug and propagates."""

from __future__ import annotations

from typing import Any

import pytest
from opensearchpy.exceptions import ConnectionError as OsConnectionError

from src.services.curation import field_coverage
from src.services.curation.autolabel.selection import count_unvalidated_remaining


class _FailingOpenSearch:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def count(self, **_: Any) -> dict[str, Any]:
        raise self._exc

    async def search(self, **_: Any) -> dict[str, Any]:
        raise self._exc


@pytest.mark.asyncio
async def test_unvalidated_count_is_minus_one_on_opensearch_failure() -> None:
    client = _FailingOpenSearch(OsConnectionError('N/A', 'down', Exception('x')))
    assert await count_unvalidated_remaining(client, 'idx', None, None, None) == -1


@pytest.mark.asyncio
async def test_unvalidated_count_propagates_a_programming_error() -> None:
    client = _FailingOpenSearch(TypeError('bug'))
    with pytest.raises(TypeError):
        await count_unvalidated_remaining(client, 'idx', None, None, None)


@pytest.mark.asyncio
async def test_field_coverage_propagates_a_programming_error() -> None:
    field_coverage.invalidate_field_coverage()
    with pytest.raises(TypeError):
        await field_coverage.field_coverage(_FailingOpenSearch(TypeError('bug')), frozenset({'a'}))


def test_pack_synonyms_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation.dataset_import import prepare
    from src.services.labeling import vlm_prompt_resolution

    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError('store unreadable')

    monkeypatch.setattr(vlm_prompt_resolution, 'active_prompt_pack', boom)
    with pytest.raises(RuntimeError):
        prepare._pack_synonyms()


@pytest.mark.asyncio
async def test_prune_box_embeddings_propagates_a_programming_error() -> None:
    from src.services.curation.region_box_embeddings import prune_box_embeddings

    class _Bug:
        async def mget(self, **_: Any) -> dict[str, Any]:
            raise TypeError('bug')

    with pytest.raises(TypeError):
        await prune_box_embeddings(_Bug(), index='idx', crop_ids=['c1'])


def test_undecodable_image_bytes_are_none_not_an_exception() -> None:
    from src.services.curation import reprocess_embed

    assert reprocess_embed._decode_image(b'not an image', 'x') is None
