"""Index creation, HNSW tuning and deletion for the visual-search indexes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from opensearchpy.exceptions import RequestError

from src.clients.opensearch.names import IndexName


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = logging.getLogger(__name__)


class IndexesMixin:
    """Index creation, HNSW tuning and deletion for the visual-search indexes."""

    client: AsyncOpenSearch
    embedding_dim: int

    # =========================================================================
    # HNSW Configuration by Index Type
    # =========================================================================

    def _get_hnsw_config(self, index_type: IndexName) -> dict:
        """
        Get optimized HNSW parameters for each index type.

        Different use cases require different quality/speed tradeoffs:
        - Global: Balanced (general scene matching)
        - Vehicles: Smaller index, faster search
        - People: Common queries, higher quality
        - Faces: Identity matching requires highest precision
        """
        configs = {
            IndexName.GLOBAL: {
                'ef_construction': 512,
                'm': 16,
                'ef_search': 256,
            },
            IndexName.VEHICLES: {
                'ef_construction': 256,
                'm': 12,
                'ef_search': 128,
            },
            IndexName.PEOPLE: {
                'ef_construction': 512,
                'm': 16,
                'ef_search': 256,
            },
            IndexName.FACES: {
                'ef_construction': 1024,
                'm': 32,
                'ef_search': 512,
            },
        }
        return configs.get(index_type, configs[IndexName.GLOBAL])

    # =========================================================================
    # Multi-Index Creation Methods
    # =========================================================================

    async def create_all_indexes(self, force_recreate: bool = False) -> dict[str, bool]:
        """
        Create all category-specific indexes.

        Args:
            force_recreate: Delete existing indexes if present

        Returns:
            Dict mapping index name to creation success
        """
        results = {}
        results[IndexName.GLOBAL.value] = await self.create_global_index(force_recreate)
        results[IndexName.VEHICLES.value] = await self.create_vehicles_index(force_recreate)
        results[IndexName.PEOPLE.value] = await self.create_people_index(force_recreate)
        results[IndexName.FACES.value] = await self.create_faces_index(force_recreate)
        results[IndexName.OCR.value] = await self.create_ocr_index(force_recreate)
        return results

    async def create_global_index(self, force_recreate: bool = False) -> bool:
        """
        Create visual_search_global index for whole image embeddings.

        Schema:
        - image_id (keyword): Unique identifier
        - image_path (keyword): File path or URL
        - global_embedding (knn_vector): 512-dim MobileCLIP embedding
        - cluster_id (integer): FAISS IVF cluster assignment
        - cluster_distance (float): Distance to cluster centroid
        - width, height (integer): Image dimensions
        - metadata (object): Flexible metadata
        - indexed_at (date): Ingestion timestamp
        - clustered_at (date): Cluster assignment timestamp
        """
        index_name = IndexName.GLOBAL.value
        hnsw = self._get_hnsw_config(IndexName.GLOBAL)

        index_body = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                    'knn': True,
                },
            },
            'mappings': {
                'properties': {
                    'image_id': {'type': 'keyword'},
                    'image_path': {'type': 'keyword'},
                    'global_embedding': {
                        'type': 'knn_vector',
                        'dimension': self.embedding_dim,
                        'method': {
                            'name': 'hnsw',
                            'space_type': 'cosinesimil',
                            'engine': 'faiss',
                            'parameters': {
                                'ef_construction': hnsw['ef_construction'],
                                'm': hnsw['m'],
                            },
                        },
                    },
                    'cluster_id': {'type': 'integer'},
                    'cluster_distance': {'type': 'float'},
                    'width': {'type': 'integer'},
                    'height': {'type': 'integer'},
                    'metadata': {'type': 'object', 'enabled': True},
                    'indexed_at': {'type': 'date'},
                    'clustered_at': {'type': 'date'},
                    # Duplicate detection fields
                    'imohash': {'type': 'keyword'},  # Fast constant-time hash (48KB sampled)
                    'file_size_bytes': {'type': 'long'},  # For additional verification
                    # Near-duplicate grouping (CLIP similarity)
                    'duplicate_group_id': {'type': 'keyword'},  # Group ID for near-duplicates
                    'is_duplicate_primary': {'type': 'boolean'},  # True = best quality in group
                    'duplicate_score': {'type': 'float'},  # Similarity to group primary
                }
            },
        }

        return await self._create_index(index_name, index_body, force_recreate)

    async def create_vehicles_index(self, force_recreate: bool = False) -> bool:
        """
        Create visual_search_vehicles index for vehicle detections.

        Classes: car(2), motorcycle(3), bus(5), truck(7), boat(8)

        Schema:
        - detection_id (keyword): Unique ID (image_id + box index)
        - image_id (keyword): Source image ID
        - image_path (keyword): File path or URL
        - embedding (knn_vector): 512-dim MobileCLIP embedding of cropped vehicle
        - cluster_id (integer): FAISS IVF cluster assignment
        - cluster_distance (float): Distance to cluster centroid
        - box (float[]): [x1, y1, x2, y2] normalized coordinates
        - class_id (integer): COCO class ID
        - class_name (keyword): Human-readable class name
        - confidence (float): Detection confidence
        - metadata (object): Flexible metadata
        - indexed_at (date): Ingestion timestamp
        - clustered_at (date): Cluster assignment timestamp
        """
        index_name = IndexName.VEHICLES.value
        hnsw = self._get_hnsw_config(IndexName.VEHICLES)

        index_body = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                    'knn': True,
                },
            },
            'mappings': {
                'properties': {
                    'detection_id': {'type': 'keyword'},
                    'image_id': {'type': 'keyword'},
                    'image_path': {'type': 'keyword'},
                    'embedding': {
                        'type': 'knn_vector',
                        'dimension': self.embedding_dim,
                        'method': {
                            'name': 'hnsw',
                            'space_type': 'cosinesimil',
                            'engine': 'faiss',
                            'parameters': {
                                'ef_construction': hnsw['ef_construction'],
                                'm': hnsw['m'],
                            },
                        },
                    },
                    'cluster_id': {'type': 'integer'},
                    'cluster_distance': {'type': 'float'},
                    'box': {'type': 'float'},
                    'class_id': {'type': 'integer'},
                    'class_name': {'type': 'keyword'},
                    'confidence': {'type': 'float'},
                    'metadata': {'type': 'object', 'enabled': True},
                    'indexed_at': {'type': 'date'},
                    'clustered_at': {'type': 'date'},
                }
            },
        }

        return await self._create_index(index_name, index_body, force_recreate)

    async def create_people_index(self, force_recreate: bool = False) -> bool:
        """
        Create visual_search_people index for person detections.

        Schema:
        - detection_id (keyword): Unique ID (image_id + box index)
        - image_id (keyword): Source image ID
        - image_path (keyword): File path or URL
        - embedding (knn_vector): 512-dim MobileCLIP embedding (appearance)
        - cluster_id (integer): FAISS IVF cluster assignment
        - cluster_distance (float): Distance to cluster centroid
        - box (float[]): [x1, y1, x2, y2] normalized coordinates
        - confidence (float): Detection confidence
        - has_face (boolean): Whether face was detected in this person
        - face_id (keyword): Link to face in visual_search_faces (optional)
        - metadata (object): Flexible metadata
        - indexed_at (date): Ingestion timestamp
        - clustered_at (date): Cluster assignment timestamp
        """
        index_name = IndexName.PEOPLE.value
        hnsw = self._get_hnsw_config(IndexName.PEOPLE)

        index_body = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                    'knn': True,
                },
            },
            'mappings': {
                'properties': {
                    'detection_id': {'type': 'keyword'},
                    'image_id': {'type': 'keyword'},
                    'image_path': {'type': 'keyword'},
                    'embedding': {
                        'type': 'knn_vector',
                        'dimension': self.embedding_dim,
                        'method': {
                            'name': 'hnsw',
                            'space_type': 'cosinesimil',
                            'engine': 'faiss',
                            'parameters': {
                                'ef_construction': hnsw['ef_construction'],
                                'm': hnsw['m'],
                            },
                        },
                    },
                    'cluster_id': {'type': 'integer'},
                    'cluster_distance': {'type': 'float'},
                    'box': {'type': 'float'},
                    'confidence': {'type': 'float'},
                    'has_face': {'type': 'boolean'},
                    'face_id': {'type': 'keyword'},
                    'metadata': {'type': 'object', 'enabled': True},
                    'indexed_at': {'type': 'date'},
                    'clustered_at': {'type': 'date'},
                }
            },
        }

        return await self._create_index(index_name, index_body, force_recreate)

    async def create_faces_index(self, force_recreate: bool = False) -> bool:
        """
        Create visual_search_faces index for face identity matching.

        Future: Will use ArcFace embeddings (512-dim, identity-based)
        Currently: Placeholder for face detection integration

        Schema:
        - face_id (keyword): Unique face ID
        - image_id (keyword): Source image ID
        - image_path (keyword): File path or URL
        - person_detection_id (keyword): Link to person in visual_search_people
        - embedding (knn_vector): 512-dim face embedding (ArcFace)
        - cluster_id (integer): FAISS IVF cluster assignment
        - cluster_distance (float): Distance to cluster centroid
        - box (float[]): [x1, y1, x2, y2] face bounding box
        - landmarks (object): Facial landmarks (5-point)
        - confidence (float): Face detection confidence
        - quality_score (float): Face quality assessment
        - person_id (keyword): Identity cluster ID (same person)
        - person_name (keyword): Optional name label
        - is_reference (boolean): Reference face for person
        - metadata (object): Flexible metadata
        - indexed_at (date): Ingestion timestamp
        - clustered_at (date): Cluster assignment timestamp
        """
        index_name = IndexName.FACES.value
        hnsw = self._get_hnsw_config(IndexName.FACES)

        index_body = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                    'knn': True,
                },
            },
            'mappings': {
                'properties': {
                    'face_id': {'type': 'keyword'},
                    'image_id': {'type': 'keyword'},
                    'image_path': {'type': 'keyword'},
                    'person_detection_id': {'type': 'keyword'},
                    'embedding': {
                        'type': 'knn_vector',
                        'dimension': self.embedding_dim,
                        'method': {
                            'name': 'hnsw',
                            'space_type': 'cosinesimil',
                            'engine': 'faiss',
                            'parameters': {
                                'ef_construction': hnsw['ef_construction'],
                                'm': hnsw['m'],
                            },
                        },
                    },
                    'cluster_id': {'type': 'integer'},
                    'cluster_distance': {'type': 'float'},
                    'box': {'type': 'float'},
                    'landmarks': {
                        'type': 'object',
                        'properties': {
                            'left_eye': {'type': 'float'},
                            'right_eye': {'type': 'float'},
                            'nose': {'type': 'float'},
                            'left_mouth': {'type': 'float'},
                            'right_mouth': {'type': 'float'},
                        },
                    },
                    'confidence': {'type': 'float'},
                    'quality_score': {'type': 'float'},
                    'person_id': {'type': 'keyword'},
                    'person_name': {'type': 'keyword'},
                    'is_reference': {'type': 'boolean'},
                    'metadata': {'type': 'object', 'enabled': True},
                    'indexed_at': {'type': 'date'},
                    'clustered_at': {'type': 'date'},
                    'thumbnail_b64': {
                        'type': 'text',
                        'index': False,  # Don't index the base64 string for search
                    },
                }
            },
        }

        return await self._create_index(index_name, index_body, force_recreate)

    async def create_ocr_index(self, force_recreate: bool = False) -> bool:
        """
        Create visual_search_ocr index for text detection and recognition.

        Schema:
        - ocr_id (keyword): Unique OCR result ID (image_id + text index)
        - image_id (keyword): Source image ID
        - image_path (keyword): File path or URL
        - text (text): Detected text with trigram analyzer for fuzzy search
        - text_raw (keyword): Exact text for keyword matching
        - box (float[]): [x1,y1,x2,y2,x3,y3,x4,y4] quadrilateral coordinates
        - box_normalized (float[]): [x1, y1, x2, y2] axis-aligned normalized
        - det_score (float): Detection confidence
        - rec_score (float): Recognition confidence
        - language (keyword): Detected language (optional)
        - metadata (object): Flexible metadata
        - indexed_at (date): Ingestion timestamp
        """
        index_name = IndexName.OCR.value

        index_body = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                    # OpenSearch rejects max_gram - min_gram > 1 unless raised.
                    'max_ngram_diff': 13,
                },
                'analysis': {
                    'analyzer': {
                        'trigram_analyzer': {
                            'type': 'custom',
                            'tokenizer': 'standard',
                            'filter': ['lowercase', 'trigram_filter'],
                        },
                    },
                    'filter': {
                        'trigram_filter': {
                            'type': 'ngram',
                            'min_gram': 2,
                            # The query is analysed with `standard` (one term
                            # per word), so a word only matches if it was
                            # indexed whole: grams must reach the longest
                            # word worth searching for.
                            'max_gram': 15,
                        },
                    },
                },
            },
            'mappings': {
                'properties': {
                    'ocr_id': {'type': 'keyword'},
                    'image_id': {'type': 'keyword'},
                    'image_path': {'type': 'keyword'},
                    'text': {
                        'type': 'text',
                        'analyzer': 'trigram_analyzer',
                        'search_analyzer': 'standard',
                    },
                    'text_raw': {'type': 'keyword'},
                    'box': {'type': 'float'},  # 8 coords: [x1,y1,x2,y2,x3,y3,x4,y4]
                    'box_normalized': {'type': 'float'},  # 4 coords: [x1,y1,x2,y2]
                    'det_score': {'type': 'float'},
                    'rec_score': {'type': 'float'},
                    'language': {'type': 'keyword'},
                    'metadata': {'type': 'object', 'enabled': True},
                    'indexed_at': {'type': 'date'},
                },
            },
        }

        return await self._create_index(index_name, index_body, force_recreate)

    async def _create_index(
        self, index_name: str, index_body: dict, force_recreate: bool = False
    ) -> bool:
        """
        Internal helper to create an index with error handling.
        """
        try:
            exists = await self.client.indices.exists(index=index_name)

            if exists:
                if force_recreate:
                    logger.info(f'Deleting existing index: {index_name}')
                    await self.client.indices.delete(index=index_name)
                else:
                    logger.info(f'Index already exists: {index_name}')
                    return True

            await self.client.indices.create(index=index_name, body=index_body)
            logger.info(f'Index created successfully: {index_name}')
            return True

        except RequestError as e:
            # Every API worker creates the indexes at startup, so losing the
            # create race to a sibling means the index exists: success.
            if e.error == 'resource_already_exists_exception':
                logger.info(f'Index already exists (created concurrently): {index_name}')
                return True
            logger.error(f'Failed to create index {index_name}: {e}')
            return False
        except Exception as e:
            logger.error(f'Failed to create index {index_name}: {e}')
            return False

    async def delete_all_indexes(self) -> dict[str, bool]:
        """Delete all visual search indexes."""
        results = {}
        for index_name in [
            IndexName.GLOBAL,
            IndexName.VEHICLES,
            IndexName.PEOPLE,
            IndexName.FACES,
            IndexName.OCR,
        ]:
            try:
                exists = await self.client.indices.exists(index=index_name.value)
                if exists:
                    await self.client.indices.delete(index=index_name.value)
                    results[index_name.value] = True
                    logger.info(f'Deleted index: {index_name.value}')
                else:
                    results[index_name.value] = True  # Already doesn't exist
            except Exception as e:
                logger.error(f'Failed to delete {index_name.value}: {e}')
                results[index_name.value] = False
        return results
