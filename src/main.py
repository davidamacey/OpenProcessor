"""
Unified Visual AI API Service.

A high-performance FastAPI service providing comprehensive visual AI capabilities:
- Object detection (YOLO11)
- Face detection and recognition (SCRFD + ArcFace)
- Image and text embeddings (MobileCLIP)
- Visual similarity search (OpenSearch k-NN)
- Data ingestion with duplicate detection
- OCR text extraction (PP-OCRv5)
- Clustering and album organization (FAISS IVF)

All inference runs through NVIDIA Triton Inference Server for optimal GPU utilization.

The application is assembled in ``src.app.factory``; this module only builds the
instance that ``uvicorn src.main:app`` serves.
"""

from src.app.factory import create_app
from src.config import get_settings
from src.core.logging import configure_logging


settings = get_settings()
configure_logging(json_logs=settings.json_logs, log_level=settings.log_level)

app = create_app()
