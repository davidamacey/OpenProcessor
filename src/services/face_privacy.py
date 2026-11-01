"""What face data a deployment still stores, and the removal of all of it (#379).

Face embeddings, person ids and person names are personal data. They live in the
deployment-wide ``visual_search_faces`` index (outside project isolation) and in the
FAISS cluster files of the ``faces`` index. Nothing here returns an id, a name or a
vector: only counts and file names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.clients.opensearch.names import IndexName
from src.config.cluster_index import ClusterIndex
from src.services.visual_search.clustering_persist import index_files


if TYPE_CHECKING:
    from pathlib import Path

_PAGE = 1000


@dataclass(frozen=True)
class FaceDataCounts:
    face_index_exists: bool
    face_docs: int
    persons: int
    named_persons: int
    thumbnails: int
    people_face_links: int
    faiss_files: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.face_index_exists
            or self.face_docs
            or self.persons
            or self.named_persons
            or self.thumbnails
            or self.people_face_links
            or self.faiss_files
        )


PEOPLE_FACE_LINK_QUERY: dict[str, Any] = {
    'bool': {
        'should': [{'exists': {'field': 'face_id'}}, {'term': {'has_face': True}}],
        'minimum_should_match': 1,
    }
}


async def _distinct_person_ids(client: Any, query: dict[str, Any]) -> int:
    """Exact distinct ``person_id`` count under ``query`` (a composite aggregation paged
    through, so a large deployment is not capped by a cardinality estimate)."""
    total = 0
    after: dict[str, Any] | None = None
    while True:
        composite: dict[str, Any] = {
            'size': _PAGE,
            'sources': [{'p': {'terms': {'field': 'person_id'}}}],
        }
        if after:
            composite['after'] = after
        resp = await client.search(
            index=IndexName.FACES.value,
            body={'size': 0, 'query': query, 'aggs': {'persons': {'composite': composite}}},
        )
        agg = resp['aggregations']['persons']
        total += len(agg['buckets'])
        after = agg.get('after_key')
        if not after or len(agg['buckets']) < _PAGE:
            return total


async def _count(client: Any, index: str, query: dict[str, Any]) -> int:
    return int((await client.count(index=index, body={'query': query}))['count'])


async def face_data_counts(client: Any, index_dir: Path) -> FaceDataCounts:
    """Count the stored face data. ``client`` is the raw ``AsyncOpenSearch``. A missing
    index counts as zero (``indices.exists``, no 404 as control flow)."""
    faces = IndexName.FACES.value
    people = IndexName.PEOPLE.value
    face_index_exists = bool(await client.indices.exists(index=faces))
    face_docs = persons = named = thumbnails = 0
    if face_index_exists:
        face_docs = await _count(client, faces, {'match_all': {}})
        persons = await _distinct_person_ids(client, {'exists': {'field': 'person_id'}})
        named = await _distinct_person_ids(
            client,
            {
                'bool': {
                    'must': [
                        {'exists': {'field': 'person_id'}},
                        {'exists': {'field': 'person_name'}},
                    ]
                }
            },
        )
        thumbnails = await _count(client, faces, {'exists': {'field': 'thumbnail_b64'}})
    links = 0
    if await client.indices.exists(index=people):
        links = await _count(client, people, PEOPLE_FACE_LINK_QUERY)
    files = [p.name for p in index_files(index_dir, ClusterIndex.FACES) if p.exists()]
    return FaceDataCounts(
        face_index_exists=face_index_exists,
        face_docs=face_docs,
        persons=persons,
        named_persons=named,
        thumbnails=thumbnails,
        people_face_links=links,
        faiss_files=files,
    )


def stored_summary(counts: FaceDataCounts) -> str:
    """One sentence for the health report and the startup log; counts only."""
    return (
        f'faces are off but face data is stored ({counts.face_docs} face docs, '
        f'{counts.persons} persons, {len(counts.faiss_files)} FAISS files, '
        f'{counts.people_face_links} people-index face links); the face routes refuse it; '
        'remove it with ./openprocessor faces purge --yes'
    )


async def stored_face_counts() -> FaceDataCounts:
    """Counts through the shared OpenSearch client and the default FAISS directory."""
    from src.config.cluster_index import FAISS_INDEX_DIR
    from src.core.dependencies import get_opensearch

    return await face_data_counts((await get_opensearch()).client, FAISS_INDEX_DIR)
