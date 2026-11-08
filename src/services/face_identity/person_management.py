"""Auto-clustering of faces into persons and person merge/rename/delete."""

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from src.clients.occ_bulk import occ_bulk_rounds
from src.clients.opensearch.names import IndexName
from src.config.settings import Settings
from src.domain.person_identity import FaceState, PersonWrite, assign_person_ids, is_person_locked
from src.services.face_identity.person_grouping import similarity_components
from src.services.inference import InferenceService


if TYPE_CHECKING:
    from src.clients.opensearch.client import OpenSearchClient


logger = logging.getLogger(__name__)


class PersonManagementMixin:
    settings: Settings
    inference: InferenceService

    if TYPE_CHECKING:

        @property
        def opensearch(self) -> OpenSearchClient: ...

        # Implemented by IdentificationMixin; declared so the mixins type-check alone.
        async def get_person_faces(self, person_id: str) -> list: ...

    # =========================================================================
    # Auto-Clustering and Person Management
    # =========================================================================

    async def _scan_faces(self) -> tuple[list[FaceState], np.ndarray, list[int]]:
        """Page every face: states, the embedding matrix and the face index of each row."""
        states: list[FaceState] = []
        row_face: list[int] = []
        chunks: list[np.ndarray] = []
        search_after = None
        while True:
            query: dict[str, Any] = {
                'size': 1000,
                'query': {'match_all': {}},
                '_source': [
                    'face_id',
                    'embedding',
                    'person_id',
                    'person_name',
                    'named_at',
                    'merged_at',
                ],
                'sort': [{'_id': 'asc'}],
            }
            if search_after:
                query['search_after'] = search_after
            response = await self.opensearch.client.search(index=IndexName.FACES.value, body=query)
            hits = response['hits']['hits']
            if not hits:
                break
            vectors = []
            for hit in hits:
                source = hit['_source']
                if source.get('embedding'):
                    row_face.append(len(states))
                    vectors.append(source['embedding'])
                states.append(
                    FaceState(
                        face_id=source.get('face_id') or hit['_id'],
                        person_id=source.get('person_id'),
                        person_name=source.get('person_name'),
                        locked=is_person_locked(source),
                    )
                )
            if vectors:
                chunks.append(np.asarray(vectors, dtype=np.float32))
            search_after = hits[-1]['sort']
        matrix = np.concatenate(chunks) if chunks else np.zeros((0, 1), dtype=np.float32)
        return states, matrix, row_face

    async def _write_assignments(
        self, writes: list[PersonWrite], scanned: dict[str, FaceState]
    ) -> tuple[int, int, int]:
        """Write person ids with OCC; a face changed since the scan is skipped, not overwritten.

        Returns ``(updated, skipped, failed)``.
        """
        by_id = {w.face_id: w for w in writes}
        timestamp = datetime.now(UTC).isoformat()
        counts = {'updated': 0, 'skipped': 0, 'failed': 0}

        def prepare(face_id: str, doc: dict[str, Any] | None) -> dict[str, Any] | None:
            seen = scanned[face_id]
            current = (doc or {}).get('_source') or {}
            if (
                doc is None
                or is_person_locked(current)
                or current.get('person_id') != seen.person_id
                or current.get('person_name') != seen.person_name
            ):
                counts['skipped'] += 1
                return None
            w = by_id[face_id]
            return {
                'person_id': w.person_id,
                'person_name': w.person_name,
                'cluster_assigned_at': timestamp,
            }

        def on_written(_face_id: str) -> None:
            counts['updated'] += 1

        def on_failed(_face_id: str, _doc: dict[str, Any] | None, _why: str) -> None:
            counts['failed'] += 1

        await occ_bulk_rounds(
            self.opensearch.client,
            index=IndexName.FACES.value,
            ids=list(by_id),
            prepare=prepare,
            on_written=on_written,
            on_failed=on_failed,
            max_retries=2,
            refresh=True,
            source_includes=['person_id', 'person_name', 'named_at', 'merged_at'],
        )
        return counts['updated'], counts['skipped'], counts['failed']

    async def person_assignments(self) -> dict[str, str]:
        """``face_id -> person_id`` for every face that has a person (ids only, no vectors)."""
        out: dict[str, str] = {}
        search_after = None
        while True:
            query: dict[str, Any] = {
                'size': 5000,
                'query': {'exists': {'field': 'person_id'}},
                '_source': ['face_id', 'person_id'],
                'sort': [{'_id': 'asc'}],
            }
            if search_after:
                query['search_after'] = search_after
            response = await self.opensearch.client.search(index=IndexName.FACES.value, body=query)
            hits = response['hits']['hits']
            if not hits:
                return out
            for hit in hits:
                source = hit['_source']
                out[source.get('face_id') or hit['_id']] = source['person_id']
            search_after = hits[-1]['sort']

    async def auto_cluster_faces(
        self,
        similarity_threshold: float = 0.7,
        min_faces_per_person: int = 2,
    ) -> dict:
        """
        Group every face into a person, keeping ids stable across runs.

        Faces with cosine similarity >= threshold are linked and each connected
        component is a person (exact search, see ``person_grouping``). A component
        below ``min_faces_per_person`` splits into one-face persons, so every face
        has a person_id; faces without an embedding are one-face persons too.
        A group keeps the stored id it overlaps most (``domain.person_identity``).
        A face whose person was named or merged is locked: it is never written,
        but still votes, and new matching faces join that person.

        Memory: the embedding matrix twice (array and index), O(n) state and one
        batch of matches. Compute is O(n^2); there is no face-count cap.

        Returns:
            dict with status, total_faces, persons_total, persons_kept, persons_new,
            persons_retired, singletons, faces_updated, faces_locked, faces_skipped
            (changed or locked since the scan) and similarity_threshold. status is
            'error' when a write failed.
        """
        try:
            states, matrix, row_face = await self._scan_faces()
            components: list[list[int]] = []
            if row_face:
                groups = await asyncio.to_thread(
                    similarity_components, matrix, similarity_threshold
                )
                components = [[row_face[r] for r in g] for g in groups]
            embedded = set(row_face)
            components.extend([i] for i in range(len(states)) if i not in embedded)

            assignment = assign_person_ids(
                states,
                components,
                min_faces=min_faces_per_person,
                mint=lambda: f'person_{uuid.uuid4().hex[:12]}',
            )
            updated, skipped, failed = await self._write_assignments(
                assignment.writes, {s.face_id: s for s in states}
            )
            result = {
                'status': 'error' if failed else 'success',
                'total_faces': len(states),
                'persons_total': assignment.persons_total,
                'persons_kept': assignment.persons_kept,
                'persons_new': assignment.persons_new,
                'persons_retired': assignment.persons_retired,
                'singletons': assignment.singletons,
                'faces_updated': updated,
                'faces_locked': assignment.faces_locked,
                'faces_skipped': skipped,
                'similarity_threshold': similarity_threshold,
            }
            if failed:
                result['error'] = f'{failed} face writes failed'
            logger.info(f'Auto-clustering complete: {result}')
            return result

        except Exception as e:
            logger.error(f'Auto-clustering failed: {e}')
            return {'status': 'error', 'total_faces': 0, 'error': str(e)}

    async def merge_persons(self, source_person_id: str, target_person_id: str) -> dict:
        """
        Merge two persons into one (move all faces from source to target).

        Both sides are stamped ``merged_at`` (the clusterer will not split them) and
        the union gets one name: the target's, else the source's.

        Args:
            source_person_id: Person ID to merge from (will be dissolved)
            target_person_id: Person ID to merge into (will receive all faces)

        Returns:
            dict with merge result
        """
        try:
            timestamp = datetime.now(UTC).isoformat()
            name = None
            for pid in (target_person_id, source_person_id):
                faces = await self.get_person_faces(pid)
                name = next((f['person_name'] for f in faces if f.get('person_name')), None)
                if name:
                    break
            response = await self.opensearch.client.update_by_query(
                index=IndexName.FACES.value,
                body={
                    'query': {'terms': {'person_id': [source_person_id, target_person_id]}},
                    'script': {
                        'source': (
                            'if (ctx._source.person_id == params.source) '
                            '{ ctx._source.merged_from = params.source; } '
                            'ctx._source.person_id = params.target; '
                            'ctx._source.person_name = params.name; '
                            'ctx._source.merged_at = params.ts'
                        ),
                        'lang': 'painless',
                        'params': {
                            'source': source_person_id,
                            'target': target_person_id,
                            'name': name,
                            'ts': timestamp,
                        },
                    },
                },
                refresh=True,
            )

            updated = response.get('updated', 0)
            logger.info(f'Merged {updated} faces from {source_person_id} to {target_person_id}')

            return {
                'status': 'success',
                'source_person_id': source_person_id,
                'target_person_id': target_person_id,
                'faces_merged': updated,
            }

        except Exception as e:
            logger.error(f'Failed to merge persons: {e}')
            return {
                'status': 'error',
                'source_person_id': source_person_id,
                'target_person_id': target_person_id,
                'error': str(e),
            }

    async def rename_person(self, person_id: str, name: str) -> dict:
        """
        Set a friendly name for a person; naming locks its current faces.

        Args:
            person_id: Person ID to rename
            name: Friendly name to assign

        Returns:
            dict with rename result
        """
        try:
            timestamp = datetime.now(UTC).isoformat()
            response = await self.opensearch.client.update_by_query(
                index=IndexName.FACES.value,
                body={
                    'query': {'term': {'person_id': person_id}},
                    'script': {
                        'source': (
                            'ctx._source.person_name = params.name; '
                            'ctx._source.named_at = params.ts'
                        ),
                        'lang': 'painless',
                        'params': {'name': name, 'ts': timestamp},
                    },
                },
                refresh=True,
            )

            updated = response.get('updated', 0)
            logger.info(f'Renamed person {person_id} ({updated} faces)')

            return {
                'status': 'success',
                'person_id': person_id,
                'person_name': name,
                'faces_updated': updated,
            }

        except Exception as e:
            logger.error(f'Failed to rename person {person_id}: {e}')
            return {
                'status': 'error',
                'person_id': person_id,
                'error': str(e),
            }

    async def delete_person(self, person_id: str, delete_faces: bool = False) -> dict:
        """
        Delete a person (optionally delete all associated faces).

        Args:
            person_id: Person ID to delete
            delete_faces: If True, delete all faces. If False, just unassign them.

        Returns:
            dict with deletion result
        """
        try:
            if delete_faces:
                # Delete all faces belonging to this person
                response = await self.opensearch.client.delete_by_query(
                    index=IndexName.FACES.value,
                    body={'query': {'term': {'person_id': person_id}}},
                    refresh=True,
                )
                deleted = response.get('deleted', 0)
                logger.info(f'Deleted person {person_id} with {deleted} faces')
                return {
                    'status': 'success',
                    'person_id': person_id,
                    'faces_deleted': deleted,
                }
            # Unassign faces (unique face-based ids) and clear the lock so they can re-cluster
            faces = await self.get_person_faces(person_id)
            unassigned = 0

            for face in faces:
                new_person_id = f'person_{face.get("face_id", uuid.uuid4().hex[:12])}'
                await self.opensearch.client.update(
                    index=IndexName.FACES.value,
                    id=face.get('_id') or face.get('face_id'),
                    body={
                        'doc': {
                            'person_id': new_person_id,
                            'person_name': None,
                            'named_at': None,
                            'merged_at': None,
                            'merged_from': None,
                        }
                    },
                    refresh=False,
                )
                unassigned += 1

            await self.opensearch.client.indices.refresh(index=IndexName.FACES.value)
            logger.info(f'Dissolved person {person_id}, {unassigned} faces now individual')

            return {
                'status': 'success',
                'person_id': person_id,
                'faces_unassigned': unassigned,
            }

        except Exception as e:
            logger.error(f'Failed to delete person {person_id}: {e}')
            return {
                'status': 'error',
                'person_id': person_id,
                'error': str(e),
            }
