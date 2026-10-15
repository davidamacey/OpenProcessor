"""Plain REST calls to the benchmark-only OpenSearch (no OpenSearch client library).

The product's guard refuses throwaway indexes and forbids raw client construction outside its
factories, so this adapter speaks HTTP like ``scripts/bench/os_write_ceiling.py`` does.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class Rest:
    """The REST calls the adapter needs, one method each; every call raises on an HTTP error."""

    def __init__(self, endpoint: str, transport: httpx.BaseTransport | None = None) -> None:
        self.base = f'http://{endpoint}'
        self.http = httpx.Client(
            timeout=7200.0,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=64),
            transport=transport or httpx.HTTPTransport(retries=3),
        )

    def call(
        self,
        method: str,
        path: str,
        body: Any = None,
        params: dict[str, Any] | None = None,
        ok: tuple[int, ...] = (),
    ) -> Any:
        headers = {'content-type': 'application/json'}
        content = (
            body.encode()
            if isinstance(body, str)
            else json.dumps(body).encode()
            if body is not None
            else None
        )
        resp = self.http.request(
            method, self.base + path, content=content, params=params, headers=headers
        )
        if resp.status_code >= 400 and resp.status_code not in ok:
            raise RuntimeError(f'{method} {path} -> {resp.status_code}: {resp.text[:300]}')
        return resp.json() if resp.content else {}

    def exists(self, index: str) -> bool:
        resp = self.http.head(f'{self.base}/{index}')
        return resp.status_code == 200

    def create(self, index: str, body: dict[str, Any]) -> None:
        self.call('PUT', f'/{index}', body)

    def delete(self, index: str) -> None:
        self.call('DELETE', f'/{index}')

    def refresh(self, index: str) -> None:
        self.call('POST', f'/{index}/_refresh')

    def put_settings(self, index: str, body: dict[str, Any]) -> None:
        self.call('PUT', f'/{index}/_settings', body)

    def forcemerge(self, index: str) -> None:
        self.call('POST', f'/{index}/_forcemerge', params={'max_num_segments': 1})

    def flush(self, index: str) -> None:
        self.call('POST', f'/{index}/_flush')

    def warmup(self, index: str) -> None:
        self.call('GET', f'/_plugins/_knn/warmup/{index}')

    def knn_stats(self) -> dict[str, Any]:
        result: dict[str, Any] = self.call('GET', '/_plugins/_knn/stats')
        return result

    def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = self.call(
            'POST', f'/{index}/_search', body, params={'request_cache': 'false'}
        )
        return result

    def count(self, index: str) -> int:
        return int(self.call('GET', f'/{index}/_count')['count'])

    def stats(self, index: str, metric: str) -> dict[str, Any]:
        result: dict[str, Any] = self.call('GET', f'/{index}/_stats/{metric}')
        return result

    def jvm(self) -> dict[str, Any]:
        result: dict[str, Any] = self.call('GET', '/_nodes/stats/jvm')['nodes']
        return result

    def segments(self, index: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self.call(
            'GET', f'/_cat/segments/{index}', params={'format': 'json', 'bytes': 'b'}
        )
        return result

    def delete_by_query(self, index: str, body: dict[str, Any]) -> None:
        self.call(
            'POST',
            f'/{index}/_delete_by_query',
            body,
            params={'refresh': 'true', 'wait_for_completion': 'true', 'conflicts': 'proceed'},
        )

    def bulk(self, body: str | list[dict[str, Any]]) -> dict[str, Any]:
        """NDJSON bulk; a list of action/source dicts is serialized one per line."""
        text = body if isinstance(body, str) else ''.join(json.dumps(x) + '\n' for x in body)
        resp = self.http.post(
            f'{self.base}/_bulk',
            content=text.encode(),
            headers={'content-type': 'application/x-ndjson'},
        )
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result

    def close(self) -> None:
        self.http.close()
