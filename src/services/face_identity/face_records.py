"""Single-face record lookup, deletion and person listing."""

import logging
from typing import TYPE_CHECKING

from src.clients.opensearch.names import IndexName
from src.config.settings import Settings
from src.services.inference import InferenceService


if TYPE_CHECKING:
    from src.clients.opensearch.client import OpenSearchClient


logger = logging.getLogger(__name__)


class FaceRecordMixin:
    settings: Settings
    inference: InferenceService

    if TYPE_CHECKING:

        @property
        def opensearch(self) -> OpenSearchClient: ...

    # =========================================================================
    # Utility Methods
    # =========================================================================

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
