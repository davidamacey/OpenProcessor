"""``QueryFakeOpenSearch`` plus what the config stores need (see ``SettingsFakeOpenSearch``)."""

from __future__ import annotations

from typing import Any

from curation.query_fakes import QueryFakeOpenSearch


class SettingsFakeOpenSearch(QueryFakeOpenSearch):
    """``QueryFakeOpenSearch`` plus what the ingest-policy store needs: a real
    ``NotFoundError`` for a missing document (the plain double raises a
    ``KeyError``, which the config-store tests read as "OpenSearch unavailable"),
    an OCC-guarded ``index`` (create / ``if_seq_no``) and upserting ``update``."""

    async def get(self, *, index: str, id: str, **kw: Any) -> dict[str, Any]:
        if id not in self.docs(index):
            from opensearchpy.exceptions import NotFoundError

            raise NotFoundError(404, f'[404] not found: {index}/{id}', {})
        return await super().get(index=index, id=id, **kw)

    async def index(
        self,
        *,
        index: str,
        id: str,
        body: dict[str, Any],
        op_type: str | None = None,
        if_seq_no: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if (op_type == 'create' and id in self.docs(index)) or (
            if_seq_no is not None and if_seq_no != self.seq.get((index, id), 1)
        ):
            raise RuntimeError('409 version_conflict')
        self.docs(index)[id] = body
        self._bump(index, id)
        self.write_calls += 1
        return {'result': 'created'}

    async def update(
        self,
        *,
        index: str,
        id: str,
        body: dict[str, Any],
        **kw: Any,
    ) -> dict[str, Any]:
        if id not in self.docs(index) and body.get('doc_as_upsert'):
            self.docs(index)[id] = {}
        return await super().update(index=index, id=id, body=body, **kw)
