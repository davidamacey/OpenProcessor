"""Person management: the faces that belong to a person identity."""

import logging

from src.clients.opensearch.names import IndexName
from src.config.settings import Settings
from src.services.inference import InferenceService


logger = logging.getLogger(__name__)


class IdentificationMixin:
    settings: Settings
    inference: InferenceService

    async def get_person_faces(self, person_id: str) -> list:
        """
        Get all faces belonging to a person identity.

        Args:
            person_id: Person identifier

        Returns:
            List of face documents (without embeddings)
        """
        try:
            response = await self.opensearch.client.search(
                index=IndexName.FACES.value,
                body={
                    'query': {'term': {'person_id': person_id}},
                    'size': 1000,  # Get all faces for person
                    'sort': [
                        {'confidence': {'order': 'desc', 'unmapped_type': 'float'}},
                    ],
                    '_source': {
                        'excludes': ['embedding'],  # Don't return large embedding
                    },
                },
            )

            faces = []
            for hit in response['hits']['hits']:
                face = hit['_source']
                face['_id'] = hit['_id']
                faces.append(face)

            return faces

        except Exception as e:
            logger.error(f'Failed to get faces for person {person_id}: {e}')
            return []
