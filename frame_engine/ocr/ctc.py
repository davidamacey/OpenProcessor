"""CTC collapse of the recogniser's per-step best class and its probability.

The recogniser build of this branch ends in an arg-max (``export/ocr_graph.py``), so a call
returns ``indices`` and ``scores`` ``[B, T]``. A line is the characters of the steps whose index
is not the blank (0) and not the previous step's index; its confidence is the mean of those
steps' probabilities, and ``char_min`` the lowest. Same rule as PaddleOCR's ``CTCLabelDecode``
(PaddleX release/3.3 ``text_recognition/processors.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class CtcDecoder:
    def __init__(self, characters: list[str]) -> None:
        #: ``blank``, the dictionary lines, then the space the model was trained with.
        self.characters = characters

    @classmethod
    def from_file(cls, path: str | Path) -> CtcDecoder:
        with Path(path).open(encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]
        return cls(['blank', *lines, ' '])

    def decode(self, indices: np.ndarray, scores: np.ndarray) -> tuple[str, float, float]:
        """``(text, mean probability, lowest probability)``; both scores 0.0 for an empty line."""
        prev = np.concatenate(([0], indices[:-1]))
        keep = (indices != 0) & (indices != prev) & (indices < len(self.characters))
        if not keep.any():
            return '', 0.0, 0.0
        kept = scores[keep]
        text = ''.join(self.characters[i] for i in indices[keep])
        return text, float(kept.mean()), float(kept.min())
