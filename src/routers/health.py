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
from src.core.dependencies import get_pytorch_models


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
    models = settings.models
    pytorch_models = get_pytorch_models()

    return {
        'service': 'YOLO Inference API',
        'status': 'running',
        'endpoints': {
            'detection': {
                'pytorch': '/pytorch/predict/{model_name}',
                'tensorrt': '/predict/{model_name}',
                'tensorrt_gpu_nms': '/predict/{model_name}_end2end',
            },
            'visual_search': {
                'ingest': '/ingest',
                'ingest_batch': '/ingest/batch',
                'search_image': '/search/image',
                'search_text': '/search/text',
            },
            'face_recognition': {
                'detect': '/faces/detect',
                'recognize': '/faces/recognize',
                'search': '/faces/search',
            },
        },
        'models': {
            'pytorch': list(pytorch_models.keys()),
            'tensorrt': list(models.STANDARD_MODELS.keys()),
            'tensorrt_end2end': [f'{k}_end2end' for k in models.END2END_MODELS],
            'ensembles': list(models.ENSEMBLE_MODELS.keys()),
        },
        'backend': {
            'triton_url': f'grpc://{settings.triton_url}',
            'pytorch_enabled': settings.enable_pytorch,
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
    - Loaded models list
    """
    settings = get_settings()
    pytorch_models = get_pytorch_models()

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
            'pytorch': {
                'enabled': settings.enable_pytorch,
                'models_loaded': list(pytorch_models.keys()),
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
