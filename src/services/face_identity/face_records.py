"""Single-face record lookup, deletion and person listing."""

import logging

from src.clients.opensearch.names import IndexName
from src.config.settings import Settings
from src.services.inference import InferenceService


logger = logging.getLogger(__name__)


class FaceRecordMixin:
    settings: Settings
    inference: InferenceService

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def get_face_by_id(self, face_id: str) -> dict | None:
        """
        Get a single face document by ID.

        Args:
            face_id: Face identifier

        Returns:
            Face document or None if not found
        """
        try:
            response = await self.opensearch.client.get(
                index=IndexName.FACES.value,
                id=face_id,
                _source={'excludes': ['embedding']},
            )
            return response['_source']
        except Exception as e:
            logger.error(f'Failed to get face {face_id}: {e}')
            return None

    async def delete_face(self, face_id: str) -> bool:
        """
        Delete a face from the index.

        Args:
            face_id: Face identifier to delete

        Returns:
            True if deleted, False otherwise
        """
        try:
            await self.opensearch.client.delete(
                index=IndexName.FACES.value,
                id=face_id,
                refresh=True,
            )
            logger.info(f'Deleted face {face_id}')
            return True
        except Exception as e:
            logger.error(f'Failed to delete face {face_id}: {e}')
            return False

    async def get_all_persons(self, limit: int = 100) -> list[dict]:
        """
        Get all unique person IDs with face counts.

        Args:
            limit: Maximum number of persons to return

        Returns:
            List of person summaries with face counts
        """
        try:
            response = await self.opensearch.client.search(
                index=IndexName.FACES.value,
                body={
                    'size': 0,
                    'aggs': {
                        'persons': {
                            'terms': {
                                'field': 'person_id',  # mapped as keyword already
                                'size': limit,
                            },
                            'aggs': {
                                'reference_face': {
                                    'top_hits': {
                                        'size': 1,
                                        'sort': [
                                            {
                                                'confidence': {
                                                    'order': 'desc',
                                                    'unmapped_type': 'float',
                                                }
                                            }
                                        ],
                                        '_source': [
                                            'face_id',
                                            'thumbnail_b64',
                                            'confidence',
                                            'person_name',
                                        ],
                                    }
                                }
                            },
                        }
                    },
                },
            )

            persons = []
            for bucket in response['aggregations']['persons']['buckets']:
                ref_hit = bucket['reference_face']['hits']['hits']
                ref_face = ref_hit[0]['_source'] if ref_hit else {}

                persons.append(
                    {
                        'person_id': bucket['key'],
                        'face_count': bucket['doc_count'],
                        'reference_face_id': ref_face.get('face_id'),
                        'reference_thumbnail': ref_face.get('thumbnail_b64'),
                        'person_name': ref_face.get('person_name'),
                    }
                )

            return persons

        except Exception as e:
            logger.error(f'Failed to get all persons: {e}')
            return []
