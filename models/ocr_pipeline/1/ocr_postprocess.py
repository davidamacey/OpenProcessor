"""PP-OCRv5 post-processing for the ocr_pipeline BLS: DB detection boxes and CTC text decoding.

Kept next to ``model.py`` (Triton loads the version directory as a flat module path) so the
numpy/OpenCV logic is importable without the Triton Python backend.
"""

import logging
from pathlib import Path

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class DBPostProcess:
    """
    Post-processing for DB text detection (simplified from RapidOCR).

    DB (Differentiable Binarization) post-processing:
    1. Threshold probability map
    2. Find contours
    3. Expand boxes (unclip)
    4. Filter by area and score
    """

    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.5,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.6,
        use_dilation: bool = True,
        score_mode: str = 'fast',
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.use_dilation = use_dilation
        self.score_mode = score_mode

        # Dilation kernel
        if use_dilation:
            self.dilation_kernel = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        else:
            self.dilation_kernel = None

    def __call__(self, pred: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Process detection output.

        Args:
            pred: [1, 1, H, W] probability map
            shape: Original image (H, W)

        Returns:
            boxes: [N, 4, 2] quadrilateral boxes
            scores: [N] detection scores
        """
        # Handle different output shapes from detection model
        if pred.ndim == 4:
            pred = pred[0, 0]  # [1, 1, H, W] -> [H, W]
        elif pred.ndim == 3:
            pred = pred[0]  # [1, H, W] -> [H, W]
        # else assume it's already [H, W]

        logger.debug(f'DBPost: pred shape={pred.shape}, range=[{pred.min():.3f}, {pred.max():.3f}]')
        src_h, src_w = shape

        # Threshold
        mask = (pred > self.thresh).astype(np.uint8) * 255
        logger.debug(f'DBPost: mask pixels > {self.thresh}: {np.sum(mask > 0)}')

        # Optional dilation
        if self.dilation_kernel is not None:
            mask = cv2.dilate(mask, self.dilation_kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        logger.debug(f'DBPost: found {len(contours)} contours')

        boxes = []
        scores = []
        filtered_reasons = {'small': 0, 'low_score': 0, 'unclip_fail': 0, 'too_small': 0}

        for contour in contours[: self.max_candidates]:
            # Get bounding box
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)

            # Calculate score
            if self.score_mode == 'fast':
                # Fast mode: use bounding box center
                x, y = int(rect[0][0]), int(rect[0][1])
                x = min(max(x, 0), pred.shape[1] - 1)
                y = min(max(y, 0), pred.shape[0] - 1)
                score = float(pred[y, x])
            else:
                # Slow mode: use polygon mean
                mask_box = np.zeros_like(pred, dtype=np.uint8)
                cv2.fillPoly(mask_box, [box.astype(np.int32)], 1)
                score = float(cv2.mean(pred, mask_box)[0])

            if score < self.box_thresh:
                filtered_reasons['low_score'] += 1
                continue

            # Unclip (expand box)
            box = self._unclip(box)
            if box is None:
                filtered_reasons['unclip_fail'] += 1
                continue

            # Verify expanded box is valid
            rect = cv2.minAreaRect(box)
            if min(rect[1]) < 3:  # Too small
                filtered_reasons['too_small'] += 1
                continue

            box = cv2.boxPoints(rect)

            # Scale to original image size
            scale_x = src_w / pred.shape[1]
            scale_y = src_h / pred.shape[0]
            box[:, 0] *= scale_x
            box[:, 1] *= scale_y

            # Clip to image bounds
            box[:, 0] = np.clip(box[:, 0], 0, src_w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, src_h - 1)

            # Order points: top-left, top-right, bottom-right, bottom-left
            box = self._order_points(box)

            boxes.append(box)
            scores.append(score)

        logger.debug(f'DBPost: {len(boxes)} boxes passed, filtered: {filtered_reasons}')

        if len(boxes) == 0:
            return np.empty((0, 4, 2), dtype=np.float32), np.empty(0, dtype=np.float32)

        return np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32)

    def _unclip(self, box: np.ndarray) -> np.ndarray | None:
        """Expand box by unclip_ratio."""
        try:
            import pyclipper

            poly = pyclipper.Pyclipper()
            poly.AddPath(box.astype(np.int32).tolist(), pyclipper.PT_SUBJECT, True)

            # Calculate expansion distance
            area = cv2.contourArea(box)
            peri = cv2.arcLength(box, True)
            if peri == 0:
                return None
            distance = area * self.unclip_ratio / peri

            expanded = poly.Execute(
                pyclipper.CT_UNION,
                pyclipper.PFT_EVENODD,
                pyclipper.PFT_EVENODD,
            )

            if len(expanded) == 0:
                return None

            offset = pyclipper.PyclipperOffset()
            offset.AddPath(expanded[0], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            result = offset.Execute(distance)

            if len(result) == 0:
                return None

            return np.array(result[0], dtype=np.float32)
        except ImportError:
            # Fallback: simple box expansion
            center = box.mean(axis=0)
            return center + (box - center) * self.unclip_ratio

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum

        d = np.diff(pts, axis=1).squeeze()
        rect[1] = pts[np.argmin(d)]  # Top-right has smallest diff
        rect[3] = pts[np.argmax(d)]  # Bottom-left has largest diff

        return rect


class CTCLabelDecode:
    """CTC decoder for text recognition."""

    def __init__(self, character_dict_path: str | Path):
        self.character = ['blank']  # CTC blank token

        with Path(character_dict_path).open('r', encoding='utf-8') as f:
            for line in f:
                char = line.strip('\n')
                self.character.append(char)

        self.character.append(' ')  # Space at end

    def __call__(self, preds: np.ndarray) -> list[tuple[str, float]]:
        """
        Decode CTC output.

        Args:
            preds: [B, T, num_chars] softmax output

        Returns:
            List of (text, confidence) tuples
        """
        results = []
        preds_idx = preds.argmax(axis=2)  # [B, T]
        preds_prob = preds.max(axis=2)  # [B, T]

        for idx, prob in zip(preds_idx, preds_prob, strict=True):
            text = ''
            conf_list = []
            prev_idx = 0

            for char_idx, char_prob in zip(idx, prob, strict=True):
                # Skip blank and repeated
                if char_idx not in (0, prev_idx) and char_idx < len(self.character):
                    text += self.character[char_idx]
                    conf_list.append(char_prob)
                prev_idx = char_idx

            conf = float(np.mean(conf_list)) if conf_list else 0.0
            results.append((text, conf))

        return results
