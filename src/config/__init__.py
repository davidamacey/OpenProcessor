"""
Configuration module for the YOLO inference service.

Provides centralized configuration using Pydantic Settings with environment variable support.
"""

from src.config.curation import CurationConfig, IndexRole, get_curation_config, index_name
from src.config.detection_profile import DetectionProfile
from src.config.region_fields import RegionFields, get_region_fields
from src.config.settings import Settings, get_settings


__all__ = [
    'CurationConfig',
    'DetectionProfile',
    'IndexRole',
    'RegionFields',
    'Settings',
    'get_curation_config',
    'get_region_fields',
    'get_settings',
    'index_name',
]
