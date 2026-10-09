"""Auto-clustering of faces into persons and person merge/rename/delete."""

import logging
import uuid
from datetime import UTC, datetime

import numpy as np

from src.clients.opensearch.names import IndexName
from src.config.settings import Settings
from src.services.inference import InferenceService


logger = logging.getLogger(__name__)


class PersonManagementMixin:
    settings: Settings
    inference: InferenceService

    # =========================================================================
    # Auto-Clustering and Person Management
    # =========================================================================

    async def auto_cluster_faces(
        self,
        similarity_threshold: float = 0.7,
        min_faces_per_person: int = 2,
        max_persons: int = 1000,
    ) -> dict:
        """
        Automatically cluster all faces into person groups using embedding similarity.

        Algorithm: Connected Components via Union-Find
        - Computes pairwise cosine similarity between all ArcFace embeddings
        - Groups faces where similarity >= threshold into connected components
        - Each connected component becomes a person group

        This is suitable for small to medium datasets (< 100k faces).
        For larger datasets, use POST /clusters/train/faces which uses
        FAISS IVF (Inverted File Index) for GPU-accelerated clustering.

        ArcFace Similarity Thresholds (cosine similarity):
        - 0.5: Loose matching (may include similar-looking different people)
        - 0.6: Moderate matching (good for photo albums)
        - 0.7: Strict matching (same person verification, recommended default)
        - 0.8: Very strict (may miss some matches due to pose/lighting)

        Args:
            similarity_threshold: Minimum cosine similarity to group faces (default 0.7)
            min_faces_per_person: Minimum faces to form a person group
            max_persons: Maximum number of persons to create

        Returns:
            dict with:
                - status: 'success' or 'error'
                - total_faces: Total faces processed
                - persons_created: Number of unique persons
                - faces_assigned: Faces assigned to persons
                - singletons: Faces without matches (own person)
        """
        try:
            # 1. Fetch all faces with embeddings
            all_faces = []
            search_after = None

            while True:
                query = {
                    'size': 1000,
                    'query': {'match_all': {}},
                    '_source': ['face_id', 'embedding', 'person_id'],
                    'sort': [{'_id': 'asc'}],
                }
                if search_after:
                    query['search_after'] = search_after

                response = await self.opensearch.client.search(
                    index=IndexName.FACES.value,
                    body=query,
                )

                hits = response['hits']['hits']
                if not hits:
                    break

                for hit in hits:
                    source = hit['_source']
                    if source.get('embedding'):
                        all_faces.append(
                            {
                                'face_id': source['face_id'],
                                'embedding': np.array(source['embedding'], dtype=np.float32),
                                'current_person_id': source.get('person_id'),
                            }
                        )

                search_after = hits[-1]['sort']

                if len(all_faces) >= 100000:  # Safety limit
                    break

            if len(all_faces) < 2:
                return {
                    'status': 'success',
                    'total_faces': len(all_faces),
                    'persons_created': len(all_faces),
                    'faces_assigned': 0,
                    'singletons': len(all_faces),
                    'message': 'Not enough faces to cluster',
                }

            # 2. Build embedding matrix
            embeddings = np.array([f['embedding'] for f in all_faces])
            n_faces = len(embeddings)

            # 3. Compute similarity matrix (cosine similarity since embeddings are L2-normalized)
            similarity_matrix = embeddings @ embeddings.T

            # 4. Greedy clustering using Union-Find
            parent = list(range(n_faces))

            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            # Group faces by similarity
            for i in range(n_faces):
                for j in range(i + 1, n_faces):
                    if similarity_matrix[i, j] >= similarity_threshold:
                        union(i, j)

            # 5. Collect clusters
            clusters: dict[int, list[int]] = {}
            for i in range(n_faces):
                root = find(i)
                if root not in clusters:
                    clusters[root] = []
                clusters[root].append(i)

            # 6. Assign person_ids and update documents
            timestamp = datetime.now(UTC).isoformat()
            persons_created = 0
            faces_assigned = 0
            singletons = 0

            for face_indices in clusters.values():
                if len(face_indices) >= min_faces_per_person:
                    # Create a person group
                    person_id = f'person_{uuid.uuid4().hex[:12]}'
                    persons_created += 1

                    if persons_created > max_persons:
                        break

                    # Update all faces in this cluster
                    for idx in face_indices:
                        face = all_faces[idx]
                        try:
                            await self.opensearch.client.update(
                                index=IndexName.FACES.value,
                                id=face['face_id'],
                                body={
                                    'doc': {
                                        'person_id': person_id,
                                        'cluster_assigned_at': timestamp,
                                    }
                                },
                                refresh=False,
                            )
                            faces_assigned += 1
                        except Exception as e:
                            logger.warning(f'Failed to update face {face["face_id"]}: {e}')
                else:
                    # Singleton faces keep their own unique person_id
                    singletons += len(face_indices)

            # Refresh index
            await self.opensearch.client.indices.refresh(index=IndexName.FACES.value)

            logger.info(
                f'Auto-clustering complete: {persons_created} persons, '
                f'{faces_assigned} faces assigned, {singletons} singletons'
            )

            return {
                'status': 'success',
                'total_faces': n_faces,
                'persons_created': persons_created,
                'faces_assigned': faces_assigned,
                'singletons': singletons,
                'similarity_threshold': similarity_threshold,
            }

        except Exception as e:
            logger.error(f'Auto-clustering failed: {e}')
            return {
                'status': 'error',
                'total_faces': 0,
                'persons_created': 0,
                'faces_assigned': 0,
                'error': str(e),
            }

    async def merge_persons(self, source_person_id: str, target_person_id: str) -> dict:
        """
        Merge two persons into one (move all faces from source to target).

        Args:
            source_person_id: Person ID to merge from (will be dissolved)
            target_person_id: Person ID to merge into (will receive all faces)

        Returns:
            dict with merge result
        """
        try:
            # Update all faces from source to target
            timestamp = datetime.now(UTC).isoformat()
            response = await self.opensearch.client.update_by_query(
                index=IndexName.FACES.value,
                body={
                    'query': {'term': {'person_id': source_person_id}},
                    'script': {
                        'source': f"ctx._source.person_id = '{target_person_id}'; "
                        f"ctx._source.merged_from = '{source_person_id}'; "
                        f"ctx._source.merged_at = '{timestamp}'",
                        'lang': 'painless',
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
        Set a friendly name for a person.

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
                        'source': f"ctx._source.person_name = '{name}'; "
                        f"ctx._source.named_at = '{timestamp}'",
                        'lang': 'painless',
                    },
                },
                refresh=True,
            )

            updated = response.get('updated', 0)
            logger.info(f'Renamed person {person_id} to "{name}" ({updated} faces)')

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
            # Unassign faces (set person_id to unique face-based IDs)
            faces = await self.get_person_faces(person_id)
            unassigned = 0

            for face in faces:
                new_person_id = f'person_{face.get("face_id", uuid.uuid4().hex[:12])}'
                await self.opensearch.client.update(
                    index=IndexName.FACES.value,
                    id=face.get('_id') or face.get('face_id'),
                    body={'doc': {'person_id': new_person_id, 'person_name': None}},
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
