"""
Ultralytics End2End Export Patch
==================================

Adds TensorRT EfficientNMS plugin support for end-to-end YOLO export.
This enables ONNX models with GPU-accelerated NMS baked directly into the graph.

Source: https://github.com/levipereira/ultralytics
Version: Based on ultralytics v8.3.18
License: AGPL-3.0

Usage:
    from ultralytics_patches import apply_end2end_patch
    apply_end2end_patch()  # Apply once before using YOLO

    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.export(
        format="onnx_trt",
        topk_all=300,
        iou_thres=0.7,
        conf_thres=0.25,
        dynamic=True,
        half=True,
        normalize_boxes=True  # Output boxes in [0, 1] range
    )

Custom Arguments:
    normalize_boxes (bool): If True, output boxes are normalized to [0, 1] range.
        Default: False (outputs pixel coordinates in 640x640 space).
        When True, boxes can be scaled to any image size by multiplying:
        box_pixels = box_normalized * [img_w, img_h, img_w, img_h]
"""

from .export_method import export_onnx_trt

__version__ = '1.0.0'
__author__ = 'Levi Pereira (original), Extracted for monkey-patch'


# ============================================================================
# Monkey Patch Application
# ============================================================================

_patch_applied = False


def apply_end2end_patch():
    """
    Apply the end2end export patch to ultralytics.

    This adds the export_onnx_trt() method to the Exporter class and
    patches the __init__ to add custom End2End arguments.

    Usage:
        from ultralytics_patches import apply_end2end_patch
        apply_end2end_patch()

        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
        model.export(format="onnx_trt", ...)
    """
    global _patch_applied

    if _patch_applied:
        print('⚠️  End2End patch already applied, skipping...')
        return

    try:
        from ultralytics.engine.exporter import Exporter
    except ImportError as e:
        raise ImportError(f'Failed to import ultralytics.engine.exporter: {e}')

    # Save original __init__
    original_init = Exporter.__init__

    # Patched __init__ that adds End2End arguments
    def patched_init(self, cfg=None, overrides=None, _callbacks=None):
        # Call original __init__ - only pass cfg if not None to avoid ultralytics 8.3+ compatibility issues
        if cfg is not None:
            original_init(self, cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        else:
            original_init(self, overrides=overrides, _callbacks=_callbacks)

        # Add End2End custom arguments with defaults if not present
        if not hasattr(self.args, 'topk_all'):
            self.args.topk_all = 300
        if not hasattr(self.args, 'iou_thres'):
            self.args.iou_thres = 0.7
        if not hasattr(self.args, 'conf_thres'):
            self.args.conf_thres = 0.25
        if not hasattr(self.args, 'class_agnostic'):
            self.args.class_agnostic = False
        if not hasattr(self.args, 'mask_resolution'):
            self.args.mask_resolution = 56
        if not hasattr(self.args, 'pooler_scale'):
            self.args.pooler_scale = 0.25
        if not hasattr(self.args, 'sampling_ratio'):
            self.args.sampling_ratio = 0
        if not hasattr(self.args, 'normalize_boxes'):
            self.args.normalize_boxes = False  # Default: pixel coordinates (640x640)

    # Apply patches
    Exporter.__init__ = patched_init
    Exporter.export_onnx_trt = export_onnx_trt

    # Patch export format list
    try:
        from ultralytics.engine import exporter

        if hasattr(exporter, 'export_formats'):
            # Add onnx_trt to supported formats
            formats = exporter.export_formats()
            if 'ONNX TensorRT' not in formats['Format'].values:
                import pandas as pd

                new_row = pd.DataFrame(
                    [
                        {
                            'Format': 'ONNX TensorRT',
                            'Argument': 'onnx_trt',
                            'Suffix': '_trt.onnx',
                            'CPU': True,
                            'GPU': True,
                        }
                    ]
                )
                exporter.export_formats = lambda: pd.concat([formats, new_row], ignore_index=True)
    except Exception as e:
        print(f'⚠️  Could not update export_formats table: {e}')

    _patch_applied = True
    print('✅ End2End TensorRT NMS patch applied successfully!')
    print("   You can now use: model.export(format='onnx_trt', ...)")


def is_patch_applied():
    """Check if the patch has been applied"""
    return _patch_applied
