"""
Visual Search Router.

Provides visual similarity search endpoints using MobileCLIP and ArcFace embeddings
with OpenSearch k-NN backend.

Endpoints:
- POST /search/image - Image-to-image similarity search
- POST /search/text - Text-to-image search (CLIP text search)
- POST /search/face - Face similarity search
- POST /search/ocr - Search images by text content
- POST /search/object - Object-level similarity (vehicles, people)

All endpoints use the VisualSearchService for OpenSearch operations and
InferenceService for embedding generation.
"""

# Importing the endpoint modules registers their routes on the shared router in
# import order, which fixes the route table order, so keep the order below
# (isort would sort it alphabetically).
from src.routers.search._router import router  # isort: skip
from src.routers.search import image_text  # isort: skip  # noqa: F401
from src.routers.search import face  # isort: skip  # noqa: F401
from src.routers.search import ocr  # isort: skip  # noqa: F401
from src.routers.search import objects  # isort: skip  # noqa: F401


__all__ = ['router']
