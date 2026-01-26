"""
Health and Monitoring Router

Provides health checks, service info, and connection pool statistics.
"""

import logging
import os

import psutil
import torch
from fastapi import APIRouter

from src.clients.triton_pool import get_client_pool_stats
from src.config import get_settings


logger = logging.getLogger(__name__)

router = APIRouter(
    tags=['Health & Monitoring'],
)


@router.get('/')
def root():
    """
    Service information endpoint.

    Returns available endpoints, models, and backend configuration.
    """
    settings = get_settings()

    return {
        'service': 'Visual AI API',
        'status': 'running',
        'endpoints': {
            'detection': '/detect',
            'faces': '/faces',
            'embed': '/embed',
            'search': '/search',
            'ingest': '/ingest',
            'analyze': '/analyze',
            'ocr': '/ocr',
            'clusters': '/clusters',
            'query': '/query',
        },
        'models': {
            'detection': settings.models.YOLO_MODEL,
            'face_detection': settings.models.FACE_DETECT_MODEL,
            'face_embedding': settings.models.ARCFACE_MODEL,
            'clip_image': settings.models.CLIP_IMAGE_MODEL,
            'clip_text': settings.models.CLIP_TEXT_MODEL,
            'ocr_detection': settings.models.OCR_DET_MODEL,
            'ocr_recognition': settings.models.OCR_REC_MODEL,
        },
        'backend': {
            'triton_url': f'grpc://{settings.triton_url}',
            'gpu_available': torch.cuda.is_available(),
        },
    }


@router.get('/health')
def health():
    """
    Health check with service status and performance metrics.

    Returns:
    - Service status
    - Triton connection status
    - OpenSearch connection status
    - GPU memory usage
    """
    settings = get_settings()

    # Process metrics
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    health_data = {
        'status': 'healthy',
        'services': {
            'triton': {
                'url': settings.triton_url,
                'protocol': 'gRPC',
                'status': 'connected',
            },
            'opensearch': {
                'url': settings.opensearch_url,
                'status': 'connected',
            },
        },
        'resources': {
            'memory_mb': round(memory_info.rss / 1024 / 1024, 2),
            'cpu_percent': process.cpu_percent(),
        },
    }

    # Add GPU metrics if available
    if torch.cuda.is_available():
        health_data['resources']['gpu'] = {
            'name': torch.cuda.get_device_name(0),
            'memory_allocated_mb': round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
            'memory_reserved_mb': round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
        }

    return health_data


@router.get('/connection_pool_info')
def connection_pool_info():
    """
    Triton gRPC connection pool statistics.

    Useful for monitoring shared client usage and A/B testing.
    """
    stats = get_client_pool_stats()

    return {
        'shared_client_pool': {
            'active': stats['active_connections'] > 0,
            'connection_count': stats['active_connections'],
            'triton_urls': stats['urls'],
        },
        'usage_info': {
            'shared_client_enabled': 'Use ?shared_client=true (default)',
            'per_request_client': 'Use ?shared_client=false for testing',
            'performance_impact': {
                'shared_mode': 'Enables batching, 400-600 RPS (recommended)',
                'per_request_mode': 'No batching, 50-100 RPS (testing only)',
            },
        },
        'testing': {
            'example_shared': "curl 'http://localhost:9600/predict/small?shared_client=true' -F 'image=@test.jpg'",
            'example_per_request': "curl 'http://localhost:9600/predict/small?shared_client=false' -F 'image=@test.jpg'",
            'check_batching': "docker compose logs triton-api | grep 'batch size'",
        },
    }
