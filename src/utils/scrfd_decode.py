"""
SCRFD face detection post-processor (CPU numpy).

Decodes raw SCRFD output tensors from Triton into boxes, scores, and 5-point
landmarks. Handles anchor generation, distance-to-bbox/kps decoding, and NMS.

SCRFD architecture:
- 3-level FPN with strides [8, 16, 32]
- 2 anchors per position at each stride
- 9 output tensors: score_{8,16,32}, bbox_{8,16,32}, kps_{8,16,32}

Reference: InsightFace SCRFD (https://github.com/deepinsight/insightface)
"""

import numpy as np


# SCRFD model constants
FPN_STRIDES = [8, 16, 32]
NUM_ANCHORS = 2
INPUT_SIZE = 640

# Output tensor names matching Triton config
SCORE_NAMES = [f'score_{s}' for s in FPN_STRIDES]
BBOX_NAMES = [f'bbox_{s}' for s in FPN_STRIDES]
KPS_NAMES = [f'kps_{s}' for s in FPN_STRIDES]
ALL_OUTPUT_NAMES = SCORE_NAMES + BBOX_NAMES + KPS_NAMES


def _generate_anchors(height: int, width: int, stride: int) -> np.ndarray:
    """
    Generate anchor centers for one FPN level.

    Args:
        height: Feature map height (input_size // stride)
        width: Feature map width (input_size // stride)
        stride: FPN stride (8, 16, or 32)

    Returns:
        Anchor centers [N*NUM_ANCHORS, 2] in pixel coordinates
    """
    # Grid of (x, y) anchor centers
    anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
    # Scale to pixel coordinates
    anchor_centers = (anchor_centers * stride).reshape(-1, 2)
    # Duplicate for NUM_ANCHORS per position
    if NUM_ANCHORS > 1:
        anchor_centers = np.stack([anchor_centers] * NUM_ANCHORS, axis=1).reshape(-1, 2)
    return anchor_centers


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """
    Decode distance predictions to bounding boxes.

    SCRFD predicts (left, top, right, bottom) distances from anchor center.

    Args:
        points: [N, 2] anchor centers (x, y)
        distance: [N, 4] predicted distances (left, top, right, bottom)

    Returns:
        [N, 4] boxes in xyxy format
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """
    Decode distance predictions to keypoints.

    SCRFD predicts (dx, dy) offsets from anchor center for each of 5 landmarks.

    Args:
        points: [N, 2] anchor centers (x, y)
        distance: [N, 10] predicted offsets (5 landmarks x 2 coords)

    Returns:
        [N, 5, 2] landmark coordinates
    """
    kps = np.zeros((len(points), 5, 2), dtype=np.float32)
    for i in range(5):
        kps[:, i, 0] = points[:, 0] + distance[:, i * 2]
        kps[:, i, 1] = points[:, 1] + distance[:, i * 2 + 1]
    return kps


def _nms(dets: np.ndarray, threshold: float = 0.4) -> list[int]:
    """
    Greedy NMS on face detections.

    Args:
        dets: [N, 5] array of (x1, y1, x2, y2, score)
        threshold: IoU threshold

    Returns:
        List of kept indices
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep


def decode_scrfd_outputs(
    net_outs: dict[str, np.ndarray],
    det_scale: float,
    input_size: int = INPUT_SIZE,
    det_thresh: float = 0.5,
    nms_thresh: float = 0.4,
    max_faces: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode SCRFD raw outputs into boxes, scores, and landmarks.

    Handles all post-processing: anchor decode, confidence filter, NMS,
    and coordinate scaling back to original image space.

    Args:
        net_outs: Dict mapping tensor names to numpy arrays.
            Expected keys: score_8, score_16, score_32,
                          bbox_8, bbox_16, bbox_32,
                          kps_8, kps_16, kps_32
        det_scale: Scale factor used when resizing original image to model input.
            All output coordinates are divided by this to get original image coords.
        input_size: Model input size (default 640)
        det_thresh: Minimum confidence threshold
        nms_thresh: NMS IoU threshold
        max_faces: Maximum number of faces to return

    Returns:
        Tuple of:
        - boxes: [M, 4] in xyxy format, original image coordinates
        - scores: [M] confidence scores
        - landmarks: [M, 5, 2] in original image coordinates
    """
    all_scores = []
    all_bboxes = []
    all_kpss = []

    for stride in FPN_STRIDES:
        # Get output tensors for this stride level
        scores_key = f'score_{stride}'
        bbox_key = f'bbox_{stride}'
        kps_key = f'kps_{stride}'

        scores_raw = net_outs[scores_key]
        bbox_preds = net_outs[bbox_key]
        kps_preds = net_outs[kps_key]

        # Strip batch dimension if present from Triton dynamic batching
        # Triton returns [B, N_anchors, D] with max_batch_size > 0
        if scores_raw.ndim == 3 and scores_raw.shape[2] == 1:
            scores_raw = scores_raw[0]
            bbox_preds = bbox_preds[0]
            kps_preds = kps_preds[0]

        # Handle output shapes:
        # Standard: [N_anchors, 1], [N_anchors, 4], [N_anchors, 10]
        if scores_raw.ndim == 2:
            h = input_size // stride
            w = input_size // stride
        else:
            raise ValueError(f'Unexpected score shape: {scores_raw.shape}')

        scores = scores_raw.flatten()

        # Generate anchors
        anchor_centers = _generate_anchors(h, w, stride)

        # Scale predictions by stride (critical!)
        bbox_preds = bbox_preds * stride
        kps_preds = kps_preds * stride

        # Filter by confidence threshold (before NMS for speed)
        pos_inds = np.where(scores >= det_thresh)[0]
        if len(pos_inds) == 0:
            continue

        pos_scores = scores[pos_inds]
        pos_bbox_preds = bbox_preds[pos_inds]
        pos_kps_preds = kps_preds[pos_inds]
        pos_anchors = anchor_centers[pos_inds]

        # Decode boxes and landmarks
        bboxes = _distance2bbox(pos_anchors, pos_bbox_preds)
        kpss = _distance2kps(pos_anchors, pos_kps_preds)

        all_scores.append(pos_scores)
        all_bboxes.append(bboxes)
        all_kpss.append(kpss)

    if not all_scores:
        return np.array([]), np.array([]), np.array([])

    # Concatenate across all strides
    scores = np.concatenate(all_scores)
    bboxes = np.concatenate(all_bboxes)
    kpss = np.concatenate(all_kpss)

    # Scale back to original image coordinates
    bboxes /= det_scale
    kpss /= det_scale

    # Sort by confidence
    order = scores.argsort()[::-1]
    scores = scores[order]
    bboxes = bboxes[order]
    kpss = kpss[order]

    # Apply NMS
    pre_det = np.hstack((bboxes, scores[:, None])).astype(np.float32)
    keep = _nms(pre_det, nms_thresh)

    # Limit results
    keep = keep[:max_faces]

    return bboxes[keep], scores[keep], kpss[keep]


def preprocess_scrfd(img: np.ndarray, input_size: int = INPUT_SIZE) -> tuple[np.ndarray, float]:
    """
    Preprocess image for SCRFD inference.

    Letterbox resize to input_size maintaining aspect ratio,
    pad with zeros, normalize with mean=127.5, std=128.

    Args:
        img: BGR image from cv2.imread, shape [H, W, 3]
        input_size: Model input size (default 640)

    Returns:
        Tuple of:
        - blob: [1, 3, input_size, input_size] FP32 normalized
        - det_scale: Scale factor for inverse transform
    """
    im_ratio = float(img.shape[0]) / img.shape[1]
    model_ratio = 1.0  # Square input

    if im_ratio > model_ratio:
        new_height = input_size
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size
        new_height = int(new_width * im_ratio)

    det_scale = float(new_height) / img.shape[0]

    import cv2

    resized = cv2.resize(img, (new_width, new_height))

    # Zero-pad to input_size x input_size (top-left aligned)
    det_img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized

    # Normalize: (pixel - 127.5) / 128.0, BGR->RGB, CHW
    blob = cv2.dnn.blobFromImage(
        det_img, 1.0 / 128.0, (input_size, input_size), (127.5, 127.5, 127.5), swapRB=True
    )

    return blob, det_scale
