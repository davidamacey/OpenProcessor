"""Fakes shared by the OCR client tests."""

from __future__ import annotations

import numpy as np


class EmptyOcrResponse:
    def as_numpy(self, name: str) -> np.ndarray:
        return {
            'num_texts': np.array([0], dtype=np.int32),
            'text_boxes': np.zeros((128, 8), np.float32),
            'text_boxes_normalized': np.zeros((128, 4), np.float32),
            'texts': np.array([b''] * 128, dtype=object),
            'text_scores': np.zeros(128, np.float32),
            'rec_scores': np.zeros(128, np.float32),
        }[name]
