"""The first create of a config, activation or keymap doc is a create-only write.

Two independent writers that both read "absent" must not both succeed: the loser gets the
same conflict error an OCC mismatch gives (OpenSearch ``op_type=create`` answers 409).
The writers are simulated as independent callers that interleave at the read (a barrier on
the first GET), not as callers sharing a lock.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from curation._fake_config_opensearch import FakeConfigOpenSearch
from src.services.config_store.index import (
    ActiveConflictError,
    RevisionConflictError,
    activate,
    save_config,
)


pytestmark = pytest.mark.unbound

INDEX = 'op_prj_alpha__configs'


class RacingClient(FakeConfigOpenSearch):
    """Both writers' first GET of ``doc_id`` returns absent before either writes."""

    def __init__(self, doc_id: str) -> None:
        super().__init__()
        self._doc_id = doc_id
        self._barrier = asyncio.Barrier(2)
        self._arrived = 0

    async def get(self, index: str, id: str, ignore: Any = None) -> dict[str, Any]:  # noqa: A002
        if id == self._doc_id and self._arrived < 2:
            self._arrived += 1
            try:
                return await super().get(index, id, ignore)
            finally:
                await self._barrier.wait()
        return await super().get(index, id, ignore)


async def _outcomes(*coros: Any) -> list[Any]:
    return await asyncio.gather(*coros, return_exceptions=True)


@pytest.mark.asyncio
async def test_two_first_save_config_one_conflicts() -> None:
    client = RacingClient('pack:wheel')
    results = await _outcomes(
        *[
            save_config(
                client,
                INDEX,
                kind='prompt_pack',
                name='wheel',
                body={'w': i},
                expected_revision=None,
            )
            for i in range(2)
        ]
    )
    assert sum(isinstance(r, dict) for r in results) == 1, results
    assert sum(isinstance(r, RevisionConflictError) for r in results) == 1, results


@pytest.mark.asyncio
async def test_two_first_activate_one_conflicts() -> None:
    client = RacingClient('activation:vlm')
    results = await _outcomes(
        *[
            activate(client, INDEX, axis='vlm', name=f'n{i}', revision=None, expected_active=None)
            for i in range(2)
        ]
    )
    assert sum(isinstance(r, dict) for r in results) == 1, results
    assert sum(isinstance(r, ActiveConflictError) for r in results) == 1, results


@pytest.mark.asyncio
async def test_control_stale_seq_no_update_still_conflicts() -> None:
    client = FakeConfigOpenSearch()
    await save_config(
        client, INDEX, kind='prompt_pack', name='wheel', body={}, expected_revision=None
    )
    # Move the doc under a writer that read the earlier seq_no.
    real_get = client.get

    async def stale_get(index: str, id: str, ignore: Any = None) -> dict[str, Any]:  # noqa: A002
        got = await real_get(index, id, ignore)
        if id == 'pack:wheel':
            await client.index(index, id, got['_source'])  # concurrent writer bumps seq_no
        return got

    client.get = stale_get  # type: ignore[method-assign]
    with pytest.raises(RevisionConflictError):
        await save_config(
            client, INDEX, kind='prompt_pack', name='wheel', body={'x': 1}, expected_revision=1
        )
