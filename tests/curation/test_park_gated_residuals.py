"""``_park_gated_residuals`` must not re-write docs already parked.

Rewriting cluster_id=-3 (PARKED_CLUSTER_ID) onto a doc that's already -3
(with cluster_subid already null) is a wasted write on every re-run of
the gate. This pins the query-shape fix rather than the full
update_by_query polling machinery.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_park_query_excludes_already_parked_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation.clustering import orchestrator as orch_mod

    captured: dict[str, Any] = {}

    async def _fake_polled(
        _client: Any, *, index: str, body: dict[str, Any], **_kw: Any
    ) -> dict[str, Any]:
        captured['body'] = body
        return {'updated': 0}

    monkeypatch.setattr(orch_mod, 'run_update_by_query_polled', _fake_polled)

    client = AsyncMock()
    n = await orch_mod._park_gated_residuals(client, max_rank=5, min_blur_ratio=None)

    assert n == 0
    must_not = captured['body']['query']['bool']['must_not']
    assert {'term': {'cluster_id': orch_mod.PARKED_CLUSTER_ID}} in must_not
