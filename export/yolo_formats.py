"""Per-format YOLO11 export steps: standard and end2end ONNX, and the matching TensorRT engines."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx
import tensorrt as trt
import torch
from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from yolo_export_settings import CONF_THRESHOLD, DEVICE, HALF, IMG_SIZE, IOU_THRESHOLD, WORKSPACE_GB
from yolo_trt_build import (
    add_optimization_profile,
    build_and_save_engine,
    enable_fp16_if_available,
    parse_onnx_model,
    setup_trt_builder,
)


if TYPE_CHECKING:
    from ultralytics import YOLO


logger = logging.getLogger(__name__)


def check_gpu_memory(required_gb: float = 4.0) -> bool:
    """Check if sufficient GPU memory is available."""
    try:
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(DEVICE).total_memory
            free_memory_gb = free_memory / (1024**3)
            if free_memory_gb < required_gb:
                logger.warning(
                    f'Low GPU memory: {free_memory_gb:.1f}GB available, {required_gb}GB recommended'
                )
            return True
        logger.warning('CUDA not available to PyTorch, but continuing (TensorRT may still work)')
        return True  # TensorRT export doesn't require torch.cuda.is_available()
    except Exception as e:
        logger.warning(f'Could not check GPU memory: {e}')
        return True  # Continue anyway


def backup_existing_file(file_path: Path, suffix: str = '.old') -> None:
    """Backup existing file if it exists."""
    if file_path.exists():
        backup_path = file_path.with_suffix(file_path.suffix + suffix)
        if backup_path.exists():
            backup_path.unlink()
        shutil.move(file_path, backup_path)
        logger.debug(f'Backed up existing file to: {backup_path}')


def export_onnx_standard(model: YOLO, config: dict[str, Any]) -> dict[str, Any]:
    """
    Export standard ONNX model (no NMS).

    Args:
        model: Loaded YOLO model
        config: Model configuration dict with triton_name, max_batch, etc.

    Returns:
        Dict with status, path, and output format info
    """
    logger.info('=' * 60)
    logger.info('[1/4] Standard ONNX Export (ONNX Runtime + TensorRT EP)')
    logger.info('=' * 60)

    try:
        logger.info('Exporting ONNX with dynamic batching...')
        logger.info('  - dynamic=True: Variable batch, height, width')
        logger.info('  - simplify=True: Optimizes graph for TensorRT')

        onnx_path = model.export(
            format='onnx',
            imgsz=IMG_SIZE,
            device=DEVICE,
            dynamic=True,
            simplify=True,
            half=HALF,
            verbose=False,
        )

        # Move to Triton model repository
        triton_name = config['triton_name']
        onnx_model_dir = Path(f'/app/models/{triton_name}/1')
        onnx_model_dir.mkdir(parents=True, exist_ok=True)

        onnx_dest = onnx_model_dir / 'model.onnx'
        backup_existing_file(onnx_dest)

        shutil.copy2(onnx_path, onnx_dest)
        logger.info(f'ONNX saved to: {onnx_dest}')
        logger.info('Output: [84, 8400] - needs CPU NMS post-processing')

        return {
            'status': 'success',
            'path': str(onnx_dest),
            'host_path': str(onnx_dest).replace('/app/models', './models'),
            'output_format': '[84, 8400] - raw detections',
        }
    except Exception as e:
        logger.exception(f'ONNX export failed: {e}')
        return {'status': 'error', 'error': str(e)}


def export_trt_standard(model: YOLO, config: dict[str, Any]) -> dict[str, Any]:
    """
    Export native TensorRT engine from standard ONNX (no NMS).

    Args:
        model: Loaded YOLO model
        config: Model configuration dict

    Returns:
        Dict with status, path, and output format info
    """
    logger.info('=' * 60)
    logger.info('[2/4] Standard TensorRT Engine Export (no NMS)')
    logger.info('=' * 60)

    try:
        triton_name = config['triton_name']
        max_batch = config['max_batch']

        # First, check if we have standard ONNX, if not export it
        onnx_path = Path(f'/app/models/{triton_name}/1/model.onnx')
        if not onnx_path.exists():
            logger.info('Standard ONNX not found, exporting first...')
            result = export_onnx_standard(model, config)
            if result.get('status') != 'success':
                return result

        logger.info(f'Using standard ONNX: {onnx_path}')
        logger.info(f'Dynamic batch: min=1, opt={max_batch // 2}, max={max_batch}')
        logger.info(f'Workspace: {WORKSPACE_GB}GB, Precision: {"FP16" if HALF else "FP32"}')

        # Check GPU memory
        check_gpu_memory(WORKSPACE_GB)

        # Setup TensorRT builder using helper
        builder, config_trt, network, trt_logger = setup_trt_builder()

        # Parse ONNX
        logger.info('Parsing ONNX model...')
        parser = trt.OnnxParser(network, trt_logger)
        if not parse_onnx_model(parser, onnx_path, uint8_input=True):
            return {'status': 'error', 'error': 'ONNX parsing failed'}

        # Set optimization profile
        logger.info('Setting optimization profile...')
        add_optimization_profile(builder, config_trt, network, max_batch)

        # Set FP16
        enable_fp16_if_available(builder, config_trt)

        # Prepare output path
        trt_model_dir = Path(f'/app/models/{triton_name}_trt/1')
        trt_model_dir.mkdir(parents=True, exist_ok=True)
        trt_dest = trt_model_dir / 'model.plan'
        backup_existing_file(trt_dest)

        # Build and save engine
        if not build_and_save_engine(builder, network, config_trt, trt_dest):
            return {'status': 'error', 'error': 'Failed to build engine'}

        logger.info('Output format: [84, 8400] - raw detections (CPU NMS needed)')

        return {
            'status': 'success',
            'path': str(trt_dest),
            'host_path': str(trt_dest).replace('/app/models', './models'),
            'output_format': '[84, 8400] - raw detections',
        }
    except Exception as e:
        logger.exception(f'TensorRT export failed: {e}')
        return {'status': 'error', 'error': str(e)}


def export_onnx_end2end(
    model: YOLO, config: dict[str, Any], normalize_boxes: bool = True
) -> dict[str, Any]:
    """
    Export ONNX with TensorRT EfficientNMS operators (GPU NMS).

    Args:
        model: Loaded YOLO model
        config: Model configuration dict
        normalize_boxes: If True, output boxes in [0,1] range; otherwise pixel coords

    Returns:
        Dict with status, path, and output format info
    """
    logger.info('=' * 60)
    logger.info('[3/4] ONNX End2End Export (with GPU NMS operators)')
    logger.info('=' * 60)

    try:
        topk = config['topk']
        box_format_str = '[0,1] range' if normalize_boxes else 'pixel coords (640x640)'

        logger.info('Exporting ONNX with EfficientNMS plugin...')
        logger.info(f'  topk_all={topk}, iou_thres={IOU_THRESHOLD}, conf_thres={CONF_THRESHOLD}')
        logger.info(f'  normalize_boxes={normalize_boxes}: {box_format_str}')

        # Create export arguments with standard ONNX format
        args = get_cfg(
            overrides={
                'format': 'onnx',
                'imgsz': IMG_SIZE,
                'dynamic': True,
                'simplify': True,
                'half': HALF,
                'device': DEVICE,
                'opset': 17,
            }
        )

        # Add custom End2End arguments
        args.topk_all = topk
        args.iou_thres = IOU_THRESHOLD
        args.conf_thres = CONF_THRESHOLD
        args.class_agnostic = True
        args.mask_resolution = 56
        args.pooler_scale = 0.25
        args.sampling_ratio = 0
        args.normalize_boxes = normalize_boxes

        # Create and configure exporter
        exporter = Exporter(cfg=args, _callbacks=model.callbacks)
        exporter.args = args
        exporter.model = model.model.to(DEVICE)  # type: ignore[union-attr]  # ultralytics types YOLO.model as str | Module | None
        if hasattr(model, 'overrides'):
            exporter.model.args = model.overrides  # type: ignore[assignment]  # overrides is a dict; Module.args is the legacy attribute

        exporter.im = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        exporter.file = Path(config['pt_file'])

        # Call export_onnx_trt directly
        logger.info('Calling patched export_onnx_trt() method...')
        export_path, _ = exporter.export_onnx_trt(prefix='ONNX TRT:')  # type: ignore[attr-defined]  # method is added by the End2End patch above

        # Move to Triton model repository
        triton_name = config['triton_name']
        onnx_model_dir = Path(f'/app/models/{triton_name}_end2end/1')
        onnx_model_dir.mkdir(parents=True, exist_ok=True)

        onnx_dest = onnx_model_dir / 'model.onnx'
        export_file = Path(export_path)

        if not export_file.exists():
            logger.error(f'Export failed, file not found: {export_file}')
            return {'status': 'error', 'error': 'Export file not found'}

        backup_existing_file(onnx_dest)
        shutil.copy2(export_file, onnx_dest)

        logger.info(f'ONNX End2End saved to: {onnx_dest}')
        logger.info('Contains: TRT::EfficientNMS_TRT operators')
        logger.info('Output: num_dets, det_boxes, det_scores, det_classes')

        # Verify NMS plugin is present
        try:
            onnx_model = onnx.load(str(onnx_dest))
            ops = [node.op_type for node in onnx_model.graph.node]
            has_nms = any('NMS' in op or 'TRT' in op for op in ops)
            if has_nms:
                logger.info('Verified: NMS plugin in ONNX graph')
            else:
                logger.warning('No NMS operators found in ONNX graph!')
        except Exception as e:
            logger.debug(f'Could not verify ONNX operators: {e}')

        box_format = '[0,1] normalized' if normalize_boxes else 'pixel coords (640x640)'
        return {
            'status': 'success',
            'path': str(onnx_dest),
            'host_path': str(onnx_dest).replace('/app/models', './models'),
            'has_nms': True,
            'output_format': 'num_dets, det_boxes, det_scores, det_classes',
            'box_format': box_format,
        }

    except Exception as e:
        logger.exception(f'ONNX End2End export failed: {e}')
        return {'status': 'error', 'error': str(e)}


def export_trt_end2end(config: dict[str, Any]) -> dict[str, Any]:
    """
    Export TensorRT engine from end2end ONNX (compiled GPU NMS).

    Args:
        config: Model configuration dict

    Returns:
        Dict with status, path, and output format info
    """
    logger.info('=' * 60)
    logger.info('[4/4] TensorRT End2End Export (compiled GPU NMS)')
    logger.info('=' * 60)

    try:
        triton_name = config['triton_name']
        max_batch = config['max_batch']

        # Check for end2end ONNX
        onnx_end2end_path = Path(f'/app/models/{triton_name}_end2end/1/model.onnx')
        if not onnx_end2end_path.exists():
            logger.error(f'End2End ONNX not found: {onnx_end2end_path}')
            logger.error('Run with --formats onnx_end2end first!')
            return {'status': 'error', 'error': 'End2End ONNX not found'}

        logger.info(f'Using end2end ONNX: {onnx_end2end_path}')
        logger.info(f'Dynamic batch: min=1, opt={max_batch // 2}, max={max_batch}')
        logger.info(f'Workspace: {WORKSPACE_GB}GB, Precision: {"FP16" if HALF else "FP32"}')

        # Check GPU memory
        check_gpu_memory(WORKSPACE_GB)

        # Setup TensorRT builder using helper
        builder, config_trt, network, trt_logger = setup_trt_builder()
        logger.info('TensorRT plugins initialized (EfficientNMS available)')

        # Parse ONNX
        logger.info('Parsing ONNX model...')
        parser = trt.OnnxParser(network, trt_logger)
        if not parse_onnx_model(parser, onnx_end2end_path, uint8_input=True):
            return {'status': 'error', 'error': 'ONNX parsing failed'}

        # Log network structure
        logger.info('Network inputs:')
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            logger.info(f'  {inp.name}: shape={inp.shape}, dtype={inp.dtype}')

        logger.info('Network outputs:')
        for i in range(network.num_outputs):
            out = network.get_output(i)
            logger.info(f'  {out.name}: shape={out.shape}, dtype={out.dtype}')

        # Set optimization profile
        logger.info('Setting optimization profile...')
        add_optimization_profile(builder, config_trt, network, max_batch)

        # Set FP16
        enable_fp16_if_available(builder, config_trt)

        # Prepare output path
        trt_model_dir = Path(f'/app/models/{triton_name}_trt_end2end/1')
        trt_model_dir.mkdir(parents=True, exist_ok=True)
        trt_dest = trt_model_dir / 'model.plan'
        backup_existing_file(trt_dest)

        # Build and save engine
        if not build_and_save_engine(builder, network, config_trt, trt_dest):
            return {'status': 'error', 'error': 'Failed to build engine'}

        logger.info('Contains: Compiled GPU NMS (EfficientNMS)')
        logger.info('Output format: num_dets, det_boxes, det_scores, det_classes')

        return {
            'status': 'success',
            'path': str(trt_dest),
            'host_path': str(trt_dest).replace('/app/models', './models'),
            'has_nms': True,
            'output_format': 'num_dets, det_boxes, det_scores, det_classes',
            'source': 'end2end_onnx',
        }
    except Exception as e:
        logger.exception(f'TensorRT End2End export failed: {e}')
        return {'status': 'error', 'error': str(e)}
