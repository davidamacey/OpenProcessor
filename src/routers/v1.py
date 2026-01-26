"""
API Version 1 Router.

Groups all API endpoints under the /v1 prefix for versioned access.
This allows for backwards-compatible API evolution.

Usage:
    /v1/detect - Object detection
    /v1/faces/recognize - Face recognition
    /v1/embed/image - Image embeddings
    etc.

Note: Non-versioned routes (e.g., /detect) are still available for
backwards compatibility but may be deprecated in future versions.
"""

from fastapi import APIRouter

from src.routers import (
    analyze_router,
    clusters_router,
    detect_router,
    embed_router,
    faces_router,
    ingest_router,
    models_router,
    ocr_router,
    persons_router,
    query_router,
    search_router,
)


# Create the v1 API router
router = APIRouter(prefix='/v1', tags=['API v1'])

# Include all capability routers under /v1
router.include_router(detect_router)  # /v1/detect
router.include_router(faces_router)  # /v1/faces
router.include_router(persons_router)  # /v1/persons
router.include_router(embed_router)  # /v1/embed
router.include_router(search_router)  # /v1/search
router.include_router(ingest_router)  # /v1/ingest
router.include_router(analyze_router)  # /v1/analyze
router.include_router(clusters_router)  # /v1/clusters
router.include_router(query_router)  # /v1/query
router.include_router(ocr_router)  # /v1/ocr
router.include_router(models_router)  # /v1/models


# Export version info
API_VERSION = '1.0.0'
API_VERSION_DATE = '2025-01'
