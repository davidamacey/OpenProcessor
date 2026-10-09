"""export_onnx_trt: the Exporter method injected by the End2End patch."""

import os

import torch
import torch.nn as nn

from .onnx_wrappers import End2End_TRT


# ============================================================================
# Export Method - Monkey Patch Target
# ============================================================================


def export_onnx_trt(self, prefix='ONNX TRT:'):
    """
    Export YOLO model to ONNX format with TensorRT EfficientNMS plugin.

    This method wraps the model with End2End_TRT to bake NMS into the ONNX graph.

    Args:
        prefix (str): Logging prefix

    Returns:
        tuple: (export_path, onnx_model)
    """
    try:
        from ultralytics.utils import colorstr, LOGGER
        from ultralytics.utils.checks import check_requirements
    except ImportError:
        # Fallback if imports fail
        def colorstr(s):
            return s

        import logging

        LOGGER = logging.getLogger(__name__)

        def check_requirements(reqs):
            pass

    requirements = ['onnx>=1.12.0']

    if self.args.simplify:
        requirements += [
            'onnxsim>=0.4.33',
            'onnxruntime-gpu' if torch.cuda.is_available() else 'onnxruntime',
        ]
    check_requirements(requirements)

    import onnx  # noqa

    # Detect model type
    try:
        from ultralytics.nn.modules import v10Detect
    except ImportError:
        # Create dummy v10Detect if not available
        class v10Detect:
            pass

    try:
        from ultralytics.models.yolo.model import SegmentationModel
    except ImportError:
        # Create dummy SegmentationModel if not available
        class SegmentationModel:
            pass

    labels = len(self.model.names)
    is_det_model = True
    v10detect = False

    for k, m in self.model.named_modules():
        if isinstance(m, v10Detect):
            v10detect = True
            break

    # Save label file
    if len(self.model.names.keys()) > 0:
        label_file = os.path.splitext(self.file)[0] + '-trt.txt'
        with open(label_file, 'w') as f_trt:
            for name in self.model.names.values():
                f_trt.write(name + '\n')
        LOGGER.info(f"{prefix} Successfully generated the label file: '{label_file}'.")

    # Get opset version
    try:
        from ultralytics.engine.exporter import get_latest_opset

        opset_version = self.args.opset or get_latest_opset()
    except (ImportError, AttributeError):
        opset_version = self.args.opset or 17

    LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__} opset {opset_version}...')

    f = os.path.splitext(self.file)[0] + '-trt.onnx'

    batch_size = 'batch'
    dynamic = self.args.dynamic
    dynamic_axes = {
        'images': {0: 'batch', 2: 'height', 3: 'width'},
    }  # variable length axes
    output_axes = {
        'num_dets': {0: 'batch'},
        'det_boxes': {0: 'batch'},
        'det_scores': {0: 'batch'},
        'det_classes': {0: 'batch'},
    }

    d = {
        'stride': int(max(self.model.stride)),
        'names': self.model.names,
        'model type': 'Segmentation' if isinstance(self.model, SegmentationModel) else 'Detection',
        'train input': f'{tuple(self.im.shape[1:])} - CHW',
        'TRT Compatibility': '8.6 or above' if self.args.class_agnostic else '8.5 or above',
    }
    if not v10detect:
        d['TRT Plugins'] = (
            'TRT_EfficientNMSX, ROIAlign'
            if isinstance(self.model, SegmentationModel)
            else 'TRT_EfficientNMS'
        )

    if not isinstance(self.model, SegmentationModel):
        is_det_model = True
        output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
        shapes = [
            batch_size,
            1,
            batch_size,
            self.args.topk_all,
            4,
            batch_size,
            self.args.topk_all,
            batch_size,
            self.args.topk_all,
        ]

    else:
        is_det_model = False
        output_axes['det_masks'] = {0: 'batch'}
        output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes', 'det_masks']
        shapes = [
            batch_size,
            1,
            batch_size,
            self.args.topk_all,
            4,
            batch_size,
            self.args.topk_all,
            batch_size,
            self.args.topk_all,
            batch_size,
            self.args.topk_all,
            self.args.mask_resolution * self.args.mask_resolution,
        ]

    dynamic_axes.update(output_axes)

    if v10detect:
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.float()
        self.model.fuse()
        for k, m in self.model.named_modules():
            if isinstance(m, v10Detect):
                m.max_det = self.args.topk_all

    # Get input size for box normalization
    input_size = int(self.im.shape[-1])  # Last dim is width (assumes square input)
    normalize_boxes = getattr(self.args, 'normalize_boxes', False)

    # Wrap model with End2End_TRT
    if v10detect:
        self.model = nn.Sequential(
            self.model,
            End2End_TRT(
                self.model,
                self.args.class_agnostic,
                self.args.topk_all,
                self.args.iou_thres,
                self.args.conf_thres,
                self.args.mask_resolution,
                self.args.pooler_scale,
                self.args.sampling_ratio,
                None,
                self.args.device,
                labels,
                is_det_model,
                v10detect,
                input_size=input_size,
                normalize_boxes=normalize_boxes,
            ),
        )
    else:
        self.model = End2End_TRT(
            self.model,
            self.args.class_agnostic,
            self.args.topk_all,
            self.args.iou_thres,
            self.args.conf_thres,
            self.args.mask_resolution,
            self.args.pooler_scale,
            self.args.sampling_ratio,
            None,
            self.args.device,
            labels,
            is_det_model,
            v10detect,
            input_size=input_size,
            normalize_boxes=normalize_boxes,
        )

    # Export to ONNX
    torch.onnx.export(
        self.model.cpu() if dynamic else self.model,  # dynamic=True only compatible with cpu
        self.im.cpu() if dynamic else self.im,
        f,
        verbose=False,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,  # Force legacy ONNX exporter (new torch.export fails with End2End models)
    )

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Add metadata
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)

    # Set output shapes
    for i in model_onnx.graph.output:
        for j in i.type.tensor_type.shape.dim:
            j.dim_param = str(shapes.pop(0))

    # Simplify
    check_requirements('onnxsim')
    try:
        import onnxsim

        LOGGER.info(f'\n{prefix} Starting to simplify ONNX...')
        model_onnx, check = onnxsim.simplify(model_onnx)
        assert check, 'assert check failed'
    except Exception as e:
        LOGGER.info(f'\n{prefix} Simplifier failure: {e}')

    onnx.save(model_onnx, f)

    # Cleanup with onnx_graphsurgeon
    check_requirements('onnx_graphsurgeon')

    LOGGER.info(f'\n{prefix} Starting to cleanup ONNX using onnx_graphsurgeon...')
    try:
        import onnx_graphsurgeon as gs

        graph = gs.import_onnx(model_onnx)
        graph = graph.cleanup().toposort()
        model_onnx = gs.export_onnx(graph)
        onnx.save(model_onnx, f)
    except Exception as e:
        LOGGER.info(f'\n{prefix} Cleanup failure: {e}')

    LOGGER.info(f'{prefix} export success ✅, saved as {f}')
    return f, model_onnx
