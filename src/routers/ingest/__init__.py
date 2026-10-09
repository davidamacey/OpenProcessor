"""
Data Ingestion Router.

Provides endpoints for ingesting images into OpenSearch with automatic
indexing to appropriate categories (global, vehicles, people, faces, ocr).

Endpoints:
- POST /ingest - Single image ingestion (high concurrency support)
- POST /ingest/batch - Batch ingest (up to 64 images)
- POST /ingest/directory - Bulk load from directory path on server

Features:
- Duplicate detection via imohash
- Near-duplicate grouping via CLIP embeddings
- Auto-routing to category indexes based on detections
- OCR text extraction and indexing
- Face detection and ArcFace embedding indexing
"""

# Importing the endpoint modules registers their routes on the shared router in
# import order, which fixes the route table order, so keep the order below
# (isort would sort it alphabetically).
from src.routers.ingest._router import router  # isort: skip
from src.routers.ingest import single  # isort: skip  # noqa: F401
from src.routers.ingest import batch  # isort: skip  # noqa: F401
from src.routers.ingest import directory  # isort: skip  # noqa: F401


__all__ = ['router']
