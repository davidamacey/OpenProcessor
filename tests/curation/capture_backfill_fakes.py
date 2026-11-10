"""A job context and an OpenSearch double for the source_backfill capture and purge steps.

The double applies, in Python, what the step's painless scripts do (fill a missing capture block,
copy the times to items, drop or round a position), so a test sees the flow, the parameters the
scripts receive and what a re-run writes. The scripts themselves run against OpenSearch in the
live harness.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from src.services.source_backfill import job


if TYPE_CHECKING:
    import pytest


class FakeBackfillOpenSearch:
    def __init__(self, images: dict[str, dict[str, Any]], items: dict[str, dict[str, Any]]) -> None:
        self.images = images
        self.items = items
        self.bulk_calls = 0
        self.by_query_calls: list[dict[str, Any]] = []
        self.searched: list[str] = []
        self.refreshes: list[bool] = []

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.searched.append(index)
        after = (body.get('search_after') or [''])[0]
        ids = [i for i in sorted(self.images) if i > after][: body['size']]
        hits = [
            {
                '_id': i,
                '_source': {
                    'image_path': self.images[i].get('image_path'),
                    'capture': {'parser': (self.images[i].get('capture') or {}).get('parser')}
                    if (self.images[i].get('capture') or {}).get('parser')
                    else {},
                },
            }
            for i in ids
        ]
        return {'hits': {'hits': hits}}

    async def bulk(self, *, body: list[dict[str, Any]], refresh: bool) -> dict[str, Any]:
        self.bulk_calls += 1
        self.refreshes.append(refresh)
        out = []
        for action, payload in zip(body[::2], body[1::2], strict=True):
            doc = self.images.get(action['update']['_id'])
            if doc is None:
                out.append({'update': {'status': 404}})
                continue
            cap = doc.get('capture')
            if cap is None or cap.get('parser') is None:
                doc['capture'] = dict(payload['script']['params']['capture'])
            out.append({'update': {'status': 200}})
        return {'errors': False, 'items': out}

    async def update_by_query(
        self, *, index: str, body: dict[str, Any], **kw: Any
    ) -> dict[str, Any]:
        self.by_query_calls.append({'index': index, 'body': body, **kw})
        script = body['script']
        params = script['params']
        changed = 0
        if 'by_image' in params:
            for item in self.items.values():
                fields = params['by_image'].get(item['image_id'])
                if fields:
                    item.update(fields)
                    changed += 1
            return {'updated': changed}
        for doc in self.images.values():
            cap = doc.get('capture') or {}
            if params.get('fields'):
                if any(f in cap for f in params['fields']):
                    for f in params['fields']:
                        cap.pop(f, None)
                    changed += 1
            elif cap.get('location') and cap.get('location_precision') != 'coarse':
                scale = params['scale']
                lat = round(cap['location']['lat'] * scale) / scale
                lon = round(cap['location']['lon'] * scale) / scale
                for f in ('altitude_m', 'location_accuracy_m', 'heading_deg'):
                    cap.pop(f, None)
                cap['location'] = {'lat': lat, 'lon': lon}
                cap['location_precision'] = 'coarse'
                changed += 1
        return {'updated': changed}


class BackfillCtx:
    def __init__(self, client: Any, steps: list[str], resume_from: dict[str, Any] | None = None):
        self.project_id = 'p'
        self.resume_from = resume_from
        self.params: dict[str, Any] = {'steps': steps}
        self._client = client
        self.checkpoints: list[dict[str, Any]] = []

    def cancelled(self) -> bool:
        return False

    def paused(self) -> bool:
        return False

    def slice_due(self) -> bool:
        return False

    async def client(self, _name: str) -> Any:
        return self._client

    async def checkpoint(self, cursor: Any) -> None:
        self.checkpoints.append(cursor)

    async def progress(self, *_args: Any, **_kw: Any) -> None:
        return None


def bind_backfill(monkeypatch: pytest.MonkeyPatch, policy: str) -> None:
    @asynccontextmanager
    async def bound_job_project(ctx: Any):
        yield ctx.project_id

    async def location_policy() -> str:
        return policy

    async def stored_policy(*, fresh: bool = False) -> tuple[str, int]:
        return policy, 1

    async def no_wait() -> None:
        return None

    monkeypatch.setattr(job, 'bound_job_project', bound_job_project)
    monkeypatch.setattr(job, 'images_index', lambda: 'images')
    monkeypatch.setattr(job, 'get_location_policy', location_policy)
    monkeypatch.setattr(job, 'read_location_policy', stored_policy)
    monkeypatch.setattr(job, 'settle', no_wait)
    monkeypatch.setattr('src.services.source_backfill.capture.images_index', lambda: 'images')
    monkeypatch.setattr('src.services.source_backfill.capture.items_index', lambda: 'items')
    monkeypatch.setattr(
        'src.services.source_backfill.location_purge.images_index', lambda: 'images'
    )
