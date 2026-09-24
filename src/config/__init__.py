"""
Configuration module for the YOLO inference service.

Provides centralized configuration using Pydantic Settings with environment variable support.
"""

from src.config.curation import (
    BACKBONE_EMBEDDING_FIELD,
    CurationConfig,
    IndexRole,
    get_curation_config,
    index_name,
)
from src.config.detection_profile import DetectionProfile
from src.config.gpu_arbiter import GpuArbiterConfig, get_gpu_arbiter_config
from src.config.region_fields import RegionFields, get_region_fields
from src.config.region_state import PENDING_STATUSES, TERMINAL_STATUSES, RegionStatus
from src.config.settings import Settings, get_settings


__all__ = [
    'BACKBONE_EMBEDDING_FIELD',
    'PENDING_STATUSES',
    'TERMINAL_STATUSES',
    'CurationConfig',
    'DetectionProfile',
    'GpuArbiterConfig',
    'IndexRole',
    'RegionFields',
    'RegionStatus',
    'Settings',
    'get_curation_config',
    'get_gpu_arbiter_config',
    'get_region_fields',
    'get_settings',
    'index_name',
]
