"""Read and delete one stored image across the visual search indexes."""

import logging

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import NotFoundError

from src.clients.opensearch.names import IndexName
from src.schemas.query import DeleteImageResponse, ImageMetadata, ImageQueryResponse


logger = logging.getLogger(__name__)


async def fetch_image_data(
    client: AsyncOpenSearch, image_id: str, include_embedding: bool
) -> ImageQueryResponse:
    """The metadata and per-index counts of one image (``not_found`` when absent)."""
    # Query global index for primary metadata
    try:
        global_doc = await client.get(
            index=IndexName.GLOBAL.value,
            id=image_id,
            _source_excludes=['global_embedding'] if not include_embedding else [],
        )
        source = global_doc['_source']
    except NotFoundError:
        # Image not found in global index
        return ImageQueryResponse(
            status='not_found',
            image_id=image_id,
            error=f'Image {image_id} not found in global index',
        )

    # Build metadata
    metadata = ImageMetadata(
        image_id=image_id,
        image_path=source.get('image_path'),
        width=source.get('width'),
        height=source.get('height'),
        indexed_at=source.get('indexed_at'),
        imohash=source.get('imohash'),
        file_size_bytes=source.get('file_size_bytes'),
        duplicate_group_id=source.get('duplicate_group_id'),
        is_duplicate_primary=source.get('is_duplicate_primary'),
        duplicate_score=source.get('duplicate_score'),
        cluster_id=source.get('cluster_id'),
        cluster_distance=source.get('cluster_distance'),
    )

    indexed_in = ['global']

    # Count vehicles for this image
    vehicles_count = 0
    try:
        vehicles_resp = await client.count(
            index=IndexName.VEHICLES.value,
            body={'query': {'term': {'image_id': image_id}}},
        )
        vehicles_count = vehicles_resp.get('count', 0)
        if vehicles_count > 0:
            indexed_in.append('vehicles')
    except NotFoundError as e:
        # Index may not exist - continue checking other indexes
        logger.debug('Could not check vehicles index: %s', e)

    # Count people for this image
    people_count = 0
    try:
        people_resp = await client.count(
            index=IndexName.PEOPLE.value,
            body={'query': {'term': {'image_id': image_id}}},
        )
        people_count = people_resp.get('count', 0)
        if people_count > 0:
            indexed_in.append('people')
    except NotFoundError as e:
        # Index may not exist - continue checking other indexes
        logger.debug('Could not check people index: %s', e)

    # Count faces for this image
    faces_count = 0
    try:
        faces_resp = await client.count(
            index=IndexName.FACES.value,
            body={'query': {'term': {'image_id': image_id}}},
        )
        faces_count = faces_resp.get('count', 0)
        if faces_count > 0:
            indexed_in.append('faces')
    except NotFoundError as e:
        # Index may not exist - continue checking other indexes
        logger.debug('Could not check faces index: %s', e)

    # Check OCR
    has_ocr = False
    try:
        ocr_resp = await client.count(
            index=IndexName.OCR.value,
            body={'query': {'term': {'image_id': image_id}}},
        )
        has_ocr = ocr_resp.get('count', 0) > 0
        if has_ocr:
            indexed_in.append('ocr')
    except NotFoundError as e:
        # Index may not exist - continue checking other indexes
        logger.debug('Could not check OCR index: %s', e)

    # Get embedding if requested
    global_embedding = None
    if include_embedding and 'global_embedding' in source:
        global_embedding = source['global_embedding']

    return ImageQueryResponse(
        status='success',
        image_id=image_id,
        metadata=metadata,
        indexed_in=indexed_in,
        num_detections=vehicles_count + people_count,
        num_faces=faces_count,
        num_vehicle_detections=vehicles_count,
        num_person_detections=people_count,
        has_ocr=has_ocr,
        global_embedding=global_embedding,
    )


async def delete_image_everywhere(client: AsyncOpenSearch, image_id: str) -> DeleteImageResponse:
    """Remove one image from every index that holds it."""
    deleted_from = []
    total_deleted = 0

    # Delete from global
    try:
        await client.delete(index=IndexName.GLOBAL.value, id=image_id)
        deleted_from.append('global')
        total_deleted += 1
    except NotFoundError as e:
        # Not found is OK - document may not exist in this index
        logger.debug('Could not delete from global index: %s', e)

    # Delete from vehicles (by query - multiple docs per image)
    try:
        result = await client.delete_by_query(
            index=IndexName.VEHICLES.value,
            body={'query': {'term': {'image_id': image_id}}},
        )
        count = result.get('deleted', 0)
        if count > 0:
            deleted_from.append('vehicles')
            total_deleted += count
    except NotFoundError as e:
        # Index may not exist or document may not be in this index - continue with other indexes
        logger.debug('Could not delete from vehicles index: %s', e)

    # Delete from people
    try:
        result = await client.delete_by_query(
            index=IndexName.PEOPLE.value,
            body={'query': {'term': {'image_id': image_id}}},
        )
        count = result.get('deleted', 0)
        if count > 0:
            deleted_from.append('people')
            total_deleted += count
    except NotFoundError as e:
        # Index may not exist or document may not be in this index - continue with other indexes
        logger.debug('Could not delete from people index: %s', e)

    # Delete from faces
    try:
        result = await client.delete_by_query(
            index=IndexName.FACES.value,
            body={'query': {'term': {'image_id': image_id}}},
        )
        count = result.get('deleted', 0)
        if count > 0:
            deleted_from.append('faces')
            total_deleted += count
    except NotFoundError as e:
        # Index may not exist or document may not be in this index - continue with other indexes
        logger.debug('Could not delete from faces index: %s', e)

    # Delete from OCR
    try:
        await client.delete(index=IndexName.OCR.value, id=image_id)
        deleted_from.append('ocr')
        total_deleted += 1
    except NotFoundError as e:
        # Index may not exist or document may not be in OCR index - continue
        logger.debug('Could not delete from OCR index: %s', e)

    if total_deleted == 0:
        return DeleteImageResponse(
            status='not_found',
            image_id=image_id,
            deleted_from=[],
            total_deleted=0,
            error=f'Image {image_id} not found in any index',
        )

    return DeleteImageResponse(
        status='success',
        image_id=image_id,
        deleted_from=deleted_from,
        total_deleted=total_deleted,
    )
