"""TensorRT custom autograd operators (EfficientNMS, EfficientNMSX, ROIAlign)."""

import torch


# ============================================================================
# TensorRT Custom Operators - EfficientNMS Plugin
# ============================================================================


class TRT_EfficientNMS_85(torch.autograd.Function):
    """TensorRT EfficientNMS operation for TensorRT 8.5+"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32
        )
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
    ):
        out = g.op(
            'TRT::EfficientNMS_TRT',
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            plugin_version_s=plugin_version,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=4,
        )
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class TRT_EfficientNMS(torch.autograd.Function):
    """TensorRT EfficientNMS operation with class-agnostic support (TensorRT 8.6+)"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
        class_agnostic=0,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32
        )
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
        class_agnostic=0,
    ):
        out = g.op(
            'TRT::EfficientNMS_TRT',
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            plugin_version_s=plugin_version,
            score_activation_i=score_activation,
            class_agnostic_i=class_agnostic,
            score_threshold_f=score_threshold,
            outputs=4,
        )
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class TRT_EfficientNMSX_85(torch.autograd.Function):
    """TensorRT EfficientNMS with indices output (for segmentation)"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32
        )
        det_indices = torch.randint(0, num_boxes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes, det_indices

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
    ):
        out = g.op(
            'TRT::EfficientNMSX_TRT',
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            plugin_version_s=plugin_version,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=5,
        )
        nums, boxes, scores, classes, det_indices = out
        return nums, boxes, scores, classes, det_indices


class TRT_EfficientNMSX(torch.autograd.Function):
    """TensorRT EfficientNMS with indices and class-agnostic support"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
        class_agnostic=0,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32
        )
        det_indices = torch.randint(0, num_boxes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes, det_indices

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
        class_agnostic=0,
    ):
        out = g.op(
            'TRT::EfficientNMSX_TRT',
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            plugin_version_s=plugin_version,
            score_activation_i=score_activation,
            class_agnostic_i=class_agnostic,
            score_threshold_f=score_threshold,
            outputs=5,
        )
        nums, boxes, scores, classes, det_indices = out
        return nums, boxes, scores, classes, det_indices


class TRT_ROIAlign(torch.autograd.Function):
    """TensorRT ROIAlign operation (for instance segmentation masks)"""

    @staticmethod
    def forward(
        ctx,
        X,
        rois,
        batch_indices,
        coordinate_transformation_mode=1,
        mode=1,  # 1- avg pooling  / 0 - max pooling
        output_height=160,
        output_width=160,
        sampling_ratio=0,
        spatial_scale=0.25,
    ):
        device = rois.device
        dtype = rois.dtype
        N, C, H, W = X.shape
        num_rois = rois.shape[0]
        return torch.randn((num_rois, C, output_height, output_width), device=device, dtype=dtype)

    @staticmethod
    def symbolic(
        g,
        X,
        rois,
        batch_indices,
        coordinate_transformation_mode=1,
        mode=1,
        output_height=160,
        output_width=160,
        sampling_ratio=0,
        spatial_scale=0.25,
    ):
        return g.op(
            'TRT::ROIAlign_TRT',
            X,
            rois,
            batch_indices,
            coordinate_transformation_mode_i=coordinate_transformation_mode,
            mode_i=mode,
            output_height_i=output_height,
            output_width_i=output_width,
            sampling_ratio_i=sampling_ratio,
            spatial_scale_f=spatial_scale,
        )
