"""ONNX wrapper modules that bake TensorRT NMS into the exported graph."""

import torch

from .trt_ops import (
    TRT_EfficientNMS,
    TRT_EfficientNMS_85,
    TRT_EfficientNMSX,
    TRT_EfficientNMSX_85,
    TRT_ROIAlign,
)


# ============================================================================
# ONNX Wrapper Modules
# ============================================================================


class ONNX_EfficientNMS_TRT(torch.nn.Module):
    """ONNX module with TensorRT NMS operation for detection models"""

    def __init__(
        self,
        class_agnostic=False,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        max_wh=None,
        device=None,
        n_classes=80,
        input_size=640,
        normalize_boxes=False,
    ):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.class_agnostic = 1 if class_agnostic else 0
        self.background_class = (-1,)
        self.box_coding = (1,)
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes = n_classes
        # Normalization: output boxes in [0, 1] range instead of pixel coordinates
        # This makes downstream processing simpler - just multiply by any target image size
        self.input_size = float(input_size)
        self.normalize_boxes = normalize_boxes

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]  # YOLO11 main output is first element
        x = x.permute(0, 2, 1)
        bboxes_x = x[..., 0:1]
        bboxes_y = x[..., 1:2]
        bboxes_w = x[..., 2:3]
        bboxes_h = x[..., 3:4]
        bboxes = torch.cat([bboxes_x, bboxes_y, bboxes_w, bboxes_h], dim=-1)
        bboxes = bboxes.unsqueeze(2)  # [n_batch, n_bboxes, 4] -> [n_batch, n_bboxes, 1, 4]
        obj_conf = x[..., 4:]
        scores = obj_conf
        if self.class_agnostic == 1:
            num_det, det_boxes, det_scores, det_classes = TRT_EfficientNMS.apply(
                bboxes,
                scores,
                self.background_class,
                self.box_coding,
                self.iou_threshold,
                self.max_obj,
                self.plugin_version,
                self.score_activation,
                self.score_threshold,
                self.class_agnostic,
            )
        else:
            num_det, det_boxes, det_scores, det_classes = TRT_EfficientNMS_85.apply(
                bboxes,
                scores,
                self.background_class,
                self.box_coding,
                self.iou_threshold,
                self.max_obj,
                self.plugin_version,
                self.score_activation,
                self.score_threshold,
            )

        # Normalize boxes to [0, 1] range if requested
        # This makes downstream processing trivial: box_pixels = box_normalized * [img_w, img_h, img_w, img_h]
        # Matches Ultralytics Boxes.xyxyn property behavior
        if self.normalize_boxes:
            det_boxes = det_boxes / self.input_size

        return num_det, det_boxes, det_scores, det_classes


class ONNX_EfficientNMSX_TRT(torch.nn.Module):
    """ONNX module with TensorRT NMS operation (with indices for segmentation)"""

    def __init__(
        self,
        class_agnostic=False,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        max_wh=None,
        device=None,
        n_classes=80,
    ):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.class_agnostic = 1 if class_agnostic else 0
        self.background_class = (-1,)
        self.box_coding = (1,)
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes = n_classes

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]  # YOLO11 main output is first element
        x = x.permute(0, 2, 1)
        bboxes_x = x[..., 0:1]
        bboxes_y = x[..., 1:2]
        bboxes_w = x[..., 2:3]
        bboxes_h = x[..., 3:4]
        bboxes = torch.cat([bboxes_x, bboxes_y, bboxes_w, bboxes_h], dim=-1)
        bboxes = bboxes.unsqueeze(2)  # [n_batch, n_bboxes, 4] -> [n_batch, n_bboxes, 1, 4]
        obj_conf = x[..., 4:]
        scores = obj_conf
        if self.class_agnostic == 1:
            num_det, det_boxes, det_scores, det_classes, det_indices = TRT_EfficientNMSX.apply(
                bboxes,
                scores,
                self.background_class,
                self.box_coding,
                self.iou_threshold,
                self.max_obj,
                self.plugin_version,
                self.score_activation,
                self.score_threshold,
                self.class_agnostic,
            )
        else:
            num_det, det_boxes, det_scores, det_classes, det_indices = TRT_EfficientNMSX_85.apply(
                bboxes,
                scores,
                self.background_class,
                self.box_coding,
                self.iou_threshold,
                self.max_obj,
                self.plugin_version,
                self.score_activation,
                self.score_threshold,
            )
        return num_det, det_boxes, det_scores, det_classes, det_indices


class ONNX_End2End_MASK_TRT(torch.nn.Module):
    """ONNX module with TensorRT NMS and ROIAlign for instance segmentation"""

    def __init__(
        self,
        class_agnostic=False,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        mask_resolution=160,
        pooler_scale=0.25,
        sampling_ratio=0,
        max_wh=None,
        device=None,
        n_classes=80,
    ):
        super().__init__()
        assert isinstance(max_wh, (int)) or max_wh is None
        self.device = device if device else torch.device('cpu')
        self.class_agnostic = 1 if class_agnostic else 0
        self.max_obj = max_obj
        self.background_class = (-1,)
        self.box_coding = (1,)
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes = n_classes
        self.mask_resolution = mask_resolution
        self.pooler_scale = pooler_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, x):
        det = x[0]
        proto = x[1]
        det = det.permute(0, 2, 1)

        bboxes_x = det[..., 0:1]
        bboxes_y = det[..., 1:2]
        bboxes_w = det[..., 2:3]
        bboxes_h = det[..., 3:4]
        bboxes = torch.cat([bboxes_x, bboxes_y, bboxes_w, bboxes_h], dim=-1)
        bboxes = bboxes.unsqueeze(2)  # [n_batch, n_bboxes, 4] -> [n_batch, n_bboxes, 1, 4]
        scores = det[..., 4 : 4 + self.n_classes]

        batch_size, nm, proto_h, proto_w = proto.shape
        total_object = batch_size * self.max_obj
        masks = det[..., 4 + self.n_classes : 4 + self.n_classes + nm]

        if self.class_agnostic == 1:
            num_det, det_boxes, det_scores, det_classes, det_indices = TRT_EfficientNMSX.apply(
                bboxes,
                scores,
                self.background_class,
                self.box_coding,
                self.iou_threshold,
                self.max_obj,
                self.plugin_version,
                self.score_activation,
                self.score_threshold,
                self.class_agnostic,
            )
        else:
            num_det, det_boxes, det_scores, det_classes, det_indices = TRT_EfficientNMSX_85.apply(
                bboxes,
                scores,
                self.background_class,
                self.box_coding,
                self.iou_threshold,
                self.max_obj,
                self.plugin_version,
                self.score_activation,
                self.score_threshold,
            )

        batch_indices = torch.ones_like(det_indices) * torch.arange(
            batch_size, device=self.device, dtype=torch.int32
        ).unsqueeze(1)
        batch_indices = batch_indices.view(total_object).to(torch.long)
        det_indices = det_indices.view(total_object).to(torch.long)
        det_masks = masks[batch_indices, det_indices]

        pooled_proto = TRT_ROIAlign.apply(
            proto,
            det_boxes.view(total_object, 4),
            batch_indices,
            1,
            1,
            self.mask_resolution,
            self.mask_resolution,
            self.sampling_ratio,
            self.pooler_scale,
        )
        pooled_proto = pooled_proto.view(
            total_object,
            nm,
            self.mask_resolution * self.mask_resolution,
        )

        det_masks = (
            torch.matmul(det_masks.unsqueeze(dim=1), pooled_proto)
            .sigmoid()
            .view(batch_size, self.max_obj, self.mask_resolution * self.mask_resolution)
        )

        return num_det, det_boxes, det_scores, det_classes, det_masks


class End2End_TRT(torch.nn.Module):
    """Wrapper module for end-to-end ONNX/TensorRT export with NMS"""

    def __init__(
        self,
        model,
        class_agnostic=False,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        mask_resolution=56,
        pooler_scale=0.25,
        sampling_ratio=0,
        max_wh=None,
        device=None,
        n_classes=80,
        is_det_model=True,
        v10detect=False,
        input_size=640,
        normalize_boxes=False,
    ):
        super().__init__()
        device = device if device else torch.device('cpu')
        assert isinstance(max_wh, (int)) or max_wh is None
        self.model = model.to(device)
        self.v10detect = v10detect

        if is_det_model and not self.v10detect:
            # Note: end2end is now a read-only property in ultralytics
            # It checks for 'one2one' attribute, so we ensure one2one is not present
            if hasattr(self.model.model[-1], 'one2one'):
                delattr(self.model.model[-1], 'one2one')
            self.patch_model = ONNX_EfficientNMS_TRT
            self.end2end = self.patch_model(
                class_agnostic,
                max_obj,
                iou_thres,
                score_thres,
                max_wh,
                device,
                n_classes,
                input_size=input_size,
                normalize_boxes=normalize_boxes,
            )
            self.end2end.eval()
        elif not is_det_model and not self.v10detect:
            # Note: end2end is now a read-only property in ultralytics
            # It checks for 'one2one' attribute, so we ensure one2one is not present
            if hasattr(self.model.model[-1], 'one2one'):
                delattr(self.model.model[-1], 'one2one')
            self.patch_model = ONNX_End2End_MASK_TRT
            self.end2end = self.patch_model(
                class_agnostic,
                max_obj,
                iou_thres,
                score_thres,
                mask_resolution,
                pooler_scale,
                sampling_ratio,
                max_wh,
                device,
                n_classes,
            )
            self.end2end.eval()
        elif self.v10detect:
            self.model.model[-1].end2end = True

    def forward(self, x):
        if not self.v10detect:
            # For YOLOv8/YOLOv11, use the end2end process
            x = self.model(x)
            x = self.end2end(x)
            return x
        else:
            # For YOLOv10, manually handle the detection outputs
            x = self.model(x)
            det_boxes = x[:, :, :4]
            det_scores = x[:, :, 4]
            det_classes = x[:, :, 5].int()
            num_dets = (x[:, :, 4] > 0.0).sum(dim=1, keepdim=True).int()
            return num_dets, det_boxes, det_scores, det_classes
