"""
FastAPI routers for the unified YOLO inference API.

Routers:
- health: Health checks and monitoring
- models: Model upload, export, and management
- detect: YOLO object detection
- embed: MobileCLIP embeddings (image, text, batch, boxes)
- faces: Face detection and recognition (YOLO11-face + ArcFace)
- persons: Person management (face clustering, naming, merging)
- ingest: Data ingestion with duplicate detection and multi-index routing
- search: Visual similarity search (image, text, face, OCR)
- analyze: Combined analysis (YOLO + faces + CLIP + OCR)
- clusters: FAISS IVF clustering and albums
- query: Data retrieval and statistics
- ocr: PP-OCRv5 text extraction
"""

from src.routers.analyze import router as analyze_router
from src.routers.clusters import router as clusters_router
from src.routers.detect import router as detect_router
from src.routers.embed import router as embed_router
from src.routers.faces import router as faces_router
from src.routers.health import router as health_router
from src.routers.ingest import router as ingest_router
from src.routers.models import router as models_router
from src.routers.ocr import router as ocr_router
from src.routers.persons import router as persons_router
from src.routers.query import router as query_router
from src.routers.search import router as search_router


__all__ = [
    'analyze_router',
    'clusters_router',
    'detect_router',
    'embed_router',
    'faces_router',
    'health_router',
    'ingest_router',
    'models_router',
    'ocr_router',
    'persons_router',
    'query_router',
    'search_router',
]
