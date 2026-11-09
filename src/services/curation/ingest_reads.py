"""Read-side queries over the images index for the ingest routes: the recent-ingest summary
(``GET /ingest/status``) and the already-ingested path lookup (``POST /ingest/path_lookup``)."""

from __future__ import annotations

from typing import Any

from src.config.curation_accessors import images_index


# OpenSearch terms-query has a default 65,536 limit per call; a client sends up to 10k paths
# and they are chunked here as a margin.
_PATH_LOOKUP_CHUNK = 10_000


async def ingest_summary(opensearch: Any, run_id: str | None) -> dict[str, Any]:
    """Counts of the ingested images grouped by source and by day (the last 14 days).

    ``run_id`` scopes everything to the images carrying that ``ingest_run_id`` (an upload run's
    own images doc field). An OpenSearch failure propagates.
    """
    query: dict[str, Any] = (
        {'term': {'ingest_run_id': run_id}} if run_id is not None else {'match_all': {}}
    )
    body = {
        'size': 0,
        'track_total_hits': True,
        'query': query,
        'aggs': {
            'by_source': {
                'terms': {'field': 'source', 'size': 64},
            },
            # Filter to the last 14 days in the query rather than
            # date-histogramming the whole index and slicing to [:14] in
            # Python -- the histogram used to run over every doc ever
            # indexed just to keep the first 14 desc-sorted buckets.
            'by_day': {
                'filter': {'range': {'indexed_at': {'gte': 'now-14d/d'}}},
                'aggs': {
                    'days': {
                        'date_histogram': {
                            'field': 'indexed_at',
                            'calendar_interval': 'day',
                            'order': {'_key': 'desc'},
                        },
                    },
                },
            },
        },
    }
    resp = await opensearch.search(index=images_index(), body=body)
    aggs = resp.get('aggregations') or {}
    return {
        'total': (resp.get('hits') or {}).get('total', {}).get('value', 0),
        'by_source': (aggs.get('by_source') or {}).get('buckets', []),
        'by_day': ((aggs.get('by_day') or {}).get('days') or {}).get('buckets', []),
    }


async def known_image_paths(opensearch: Any, image_paths: list[str]) -> dict[str, str]:
    """The requested paths already ingested, mapped to their ``image_id``.

    Matches ``image_path`` OR ``source_identifier`` (both mapped keyword on the images index): a
    byte-upload ingest's client identifier is recorded as ``source_identifier`` now that
    ``image_path`` is the server-persisted path. A document found via ``source_identifier`` keys
    under that identifier, not under the persisted path the caller does not know.
    """
    result: dict[str, str] = {}
    for i in range(0, len(image_paths), _PATH_LOOKUP_CHUNK):
        chunk = image_paths[i : i + _PATH_LOOKUP_CHUNK]
        resp = await opensearch.search(
            index=images_index(),
            body={
                'size': len(chunk),
                '_source': ['image_id', 'image_path', 'source_identifier'],
                'query': {
                    'bool': {
                        'should': [
                            {'terms': {'image_path': chunk}},
                            {'terms': {'source_identifier': chunk}},
                        ],
                        'minimum_should_match': 1,
                    }
                },
            },
        )
        chunk_set = set(chunk)
        for hit in resp.get('hits', {}).get('hits', []):
            src = hit.get('_source') or {}
            iid = src.get('image_id')
            if not iid:
                continue
            for field in ('image_path', 'source_identifier'):
                p = src.get(field)
                if p and p in chunk_set:
                    result[p] = iid
    return result
