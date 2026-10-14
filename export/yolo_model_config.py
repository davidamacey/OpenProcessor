"""Model configuration for the YOLO11 exporter: config-file loading, custom-model parsing and class labels."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml


if TYPE_CHECKING:
    from ultralytics import YOLO


logger = logging.getLogger(__name__)


def load_config_file(config_path: str) -> dict[str, dict[str, Any]]:
    """
    Load model configurations from a YAML file.

    Expected YAML format:
    ```yaml
    models:
      my_custom_model:
        pt_file: /path/to/model.pt
        triton_name: my_model_name  # optional, auto-generated if not provided
        max_batch: 32  # optional, default 32
        topk: 300  # optional, default 300
        num_classes: 80  # optional, auto-detected from model
        class_names:  # optional, auto-detected from model
          - person
          - car
          - ...
    ```

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary of model configurations
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with Path(path).open() as f:
        config = yaml.safe_load(f)

    if 'models' not in config:
        raise ValueError("Config file must have a 'models' section")

    models = {}
    for model_id, model_config in config['models'].items():
        if 'pt_file' not in model_config:
            raise ValueError(f"Model '{model_id}' must have 'pt_file' specified")

        # Set defaults
        models[model_id] = {
            'pt_file': model_config['pt_file'],
            'triton_name': model_config.get('triton_name', _generate_triton_name(model_id)),
            'max_batch': model_config.get('max_batch', 32),
            'topk': model_config.get('topk', 300),
        }

        # Optional class configuration
        if 'num_classes' in model_config:
            models[model_id]['num_classes'] = model_config['num_classes']
        if 'class_names' in model_config:
            models[model_id]['class_names'] = model_config['class_names']

    logger.info(f'Loaded {len(models)} model(s) from config: {config_path}')
    return models


def parse_custom_model(custom_model_arg: str) -> tuple[str, dict[str, Any]]:
    """
    Parse a custom model argument string.

    Format: path/to/model.pt[:name][:max_batch]
    Examples:
      - /path/to/my_model.pt                  -> auto name, batch=32
      - /path/to/my_model.pt:custom_name      -> custom_name, batch=32
      - /path/to/my_model.pt:custom_name:64   -> custom_name, batch=64
      - /path/to/my_model.pt::16              -> auto name, batch=16

    Args:
        custom_model_arg: Custom model specification string

    Returns:
        Tuple of (model_id, config_dict)
    """
    parts = custom_model_arg.split(':')
    pt_file = parts[0]

    if not Path(pt_file).exists():
        # Try with /app prefix for container paths
        container_path = f'/app/{pt_file.lstrip("/")}'
        if Path(container_path).exists():
            pt_file = container_path
        else:
            raise FileNotFoundError(f'Model file not found: {parts[0]}')

    # Extract model ID from filename
    model_id = Path(pt_file).stem

    # Parse optional name
    triton_name = parts[1] if len(parts) > 1 and parts[1] else _generate_triton_name(model_id)

    # Parse optional max_batch
    max_batch = int(parts[2]) if len(parts) > 2 and parts[2] else 32

    config = {
        'pt_file': pt_file,
        'triton_name': triton_name,
        'max_batch': max_batch,
        'topk': 300,
    }

    logger.info(f'Custom model: {model_id} -> {triton_name} (batch={max_batch})')
    return model_id, config


def _generate_triton_name(model_id: str) -> str:
    """
    Generate a Triton-compatible model name from model ID.

    Converts names like 'yolo11s' to 'yolov11_small', 'my-custom-model' to 'my_custom_model'.
    """
    # Known YOLO11 size mappings
    size_map = {
        'n': 'nano',
        's': 'small',
        'm': 'medium',
        'l': 'large',
        'x': 'xlarge',
    }

    # Check for yolo11X pattern
    match = re.match(r'yolo11([nsmxl])$', model_id.lower())
    if match:
        size = size_map.get(match.group(1), match.group(1))
        return f'yolov11_{size}'

    # General cleanup: replace hyphens with underscores, lowercase
    name = model_id.lower().replace('-', '_')

    # Remove .pt extension if present
    if name.endswith('.pt'):
        name = name[:-3]

    return name


def extract_class_names(model: YOLO) -> list[str]:
    """
    Extract class names from a loaded YOLO model.

    Args:
        model: Loaded YOLO model instance

    Returns:
        List of class names in order (index = class_id)
    """
    try:
        # YOLO models store class names in model.names
        if hasattr(model, 'names'):
            names = model.names
            if isinstance(names, dict):
                # Convert dict {0: 'person', 1: 'bicycle', ...} to list
                max_idx = max(names.keys())
                return [names.get(i, f'class_{i}') for i in range(max_idx + 1)]
            if isinstance(names, list):
                return names

        # Fallback: check model.model.names
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            names = model.model.names  # type: ignore[union-attr]  # ultralytics types YOLO.model as str | Module | None
            if isinstance(names, dict):
                max_idx = max(names.keys())
                return [names.get(i, f'class_{i}') for i in range(max_idx + 1)]
            if isinstance(names, list):
                return names

        logger.warning('Could not extract class names from model')
        return []
    except Exception as e:
        logger.warning(f'Error extracting class names: {e}')
        return []


def save_labels_file(class_names: list[str], model_dir: Path) -> Path | None:
    """
    Save class names to labels.txt file in model directory.

    Args:
        class_names: List of class names
        model_dir: Model directory path (e.g., /app/models/yolov11_small_trt)

    Returns:
        Path to saved labels file, or None if no classes to save
    """
    if not class_names:
        return None

    labels_path = model_dir / 'labels.txt'
    with Path(labels_path).open('w') as f:
        for name in class_names:
            f.write(f'{name}\n')

    logger.info(f'Saved {len(class_names)} class names to: {labels_path}')
    return labels_path
