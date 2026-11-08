"""A bulk-only fake: records every update line and applies the fence like the script does."""

from __future__ import annotations

from typing import Any


class FakeOpenSearch:
    def __init__(self, fail_ids: set[str] | None = None) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.calls = 0
        self.fail_ids = fail_ids or set()

    async def bulk(
        self,
        *,
        body: list[dict[str, Any]],
        refresh: bool = False,  # noqa: ARG002 - the client's signature
    ) -> dict[str, Any]:
        self.calls += 1
        items = []
        for action, payload in zip(body[::2], body[1::2], strict=True):
            doc_id = action['update']['_id']
            if doc_id in self.fail_ids:
                items.append({'update': {'_id': doc_id, 'error': {'type': 'boom'}}})
                continue
            params = payload['script']['params']
            doc = self.docs.setdefault(doc_id, {})
            stored = doc.get('cluster_partition_id')
            if stored is not None and stored > params['pid']:  # the script's compareTo fence
                items.append({'update': {'_id': doc_id, 'result': 'noop'}})
                continue
            doc.update(
                cluster_partition_id=params['pid'],
                cluster_uid=params['uid'],
                cluster_state=params['state'],
                cluster_score=params['score'],
                cluster_margin=params['margin'],
                cluster_assigned_by=params['by'],
            )
            items.append({'update': {'_id': doc_id, 'result': 'updated'}})
        return {'errors': any('error' in i['update'] for i in items), 'items': items}
