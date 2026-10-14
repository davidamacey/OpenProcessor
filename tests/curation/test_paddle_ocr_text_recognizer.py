"""``PaddleOcrTextRecognizer``: the ocr_pipeline BLS wrapper and the text-hint rules."""

from __future__ import annotations

import dataclasses
import io
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from _region_profile_fixture import NEUTRAL_REGION_PROFILE
from PIL import Image


# =============================================================================
# PaddleOcrTextRecognizer (ocr_pipeline BLS wrapper)
# =============================================================================


def _ocr_pipeline_mock_result(
    *,
    n: int,
    boxes: list[list[float]],
    texts: list[str],
    det_scores: list[float],
    rec_scores: list[float],
) -> Any:
    """Build a fake Triton result matching the ocr_pipeline output spec.

    The wrapper pads to 128 entries the same way the BLS does; only the
    first ``n`` slots carry real data.
    """
    pad_box = [0.0, 0.0, 0.0, 0.0]
    boxes_full = boxes + [pad_box] * (128 - len(boxes))
    texts_full = texts + [''] * (128 - len(texts))
    det_full = det_scores + [0.0] * (128 - len(det_scores))
    rec_full = rec_scores + [0.0] * (128 - len(rec_scores))

    table = {
        'num_texts': np.asarray([n], dtype=np.int32),
        'text_boxes_normalized': np.asarray(boxes_full, dtype=np.float32),
        'texts': np.asarray([t.encode('utf-8') for t in texts_full], dtype=object),
        'text_scores': np.asarray(det_full, dtype=np.float32),
        'rec_scores': np.asarray(rec_full, dtype=np.float32),
    }
    m = MagicMock()
    m.as_numpy = table.get
    return m


def _jpeg_bytes(w: int = 64, h: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (w, h), (128, 128, 128)).save(buf, 'JPEG')
    return buf.getvalue()


LETTERS_AND_DIGITS_PROFILE = dataclasses.replace(
    NEUTRAL_REGION_PROFILE, text_hint_require_letters_and_digits=True
)


def _text_region(text: str, profile: Any) -> Any:
    from src.services.detection.cascade_detect.ocr_recognizer import OcrRegion

    return OcrRegion(
        bbox_norm=(0.10, 0.40, 0.45, 0.50),
        text=text,
        text_raw=text,
        det_score=0.95,
        rec_score=0.95,
        profile=profile,
    )


class TestTextHintLettersAndDigitsRule:
    @pytest.mark.parametrize('text', ['1234567', 'ABCDEF', 'AB12CD'])
    def test_off_by_default(self, text: str) -> None:
        assert NEUTRAL_REGION_PROFILE.text_hint_require_letters_and_digits is False
        assert _text_region(text, NEUTRAL_REGION_PROFILE).is_region_text_candidate

    @pytest.mark.parametrize(
        ('text', 'expected'), [('1234567', False), ('ABCDEF', False), ('AB12CD', True)]
    )
    def test_applies_when_enabled(self, text: str, expected: bool) -> None:
        assert _text_region(text, LETTERS_AND_DIGITS_PROFILE).is_region_text_candidate is expected


class TestPaddleOcrTextRecognizer:
    @pytest.mark.asyncio
    async def test_canonicalizes_and_filters(self) -> None:
        from src.services.detection.cascade_detect.ocr_recognizer import PaddleOcrTextRecognizer

        pool = MagicMock()
        pool.infer = AsyncMock(
            return_value=_ocr_pipeline_mock_result(
                n=3,
                boxes=[
                    [0.10, 0.40, 0.50, 0.50],  # plate-shaped, plate text
                    [0.05, 0.05, 0.95, 0.10],  # bumper sticker, plate-shaped, junk
                    [0.20, 0.60, 0.22, 0.62],  # tiny, non-plate-shape
                ],
                texts=['Abc-1234', 'COOLBUMPER!!!', '??'],
                det_scores=[0.90, 0.85, 0.70],
                rec_scores=[0.95, 0.80, 0.60],
            )
        )
        rec = PaddleOcrTextRecognizer(pool, NEUTRAL_REGION_PROFILE)
        regions = await rec.detect_regions(_jpeg_bytes())
        # Third entry's '??' canonicalizes to an empty string (no
        # alphanumerics survive the filter), so it's dropped — left
        # with the plate row and the bumper sticker.
        assert len(regions) == 2
        # Canonicalization: uppercased, single-spaced, kept dash.
        assert regions[0].text == 'ABC-1234'
        # text_raw preserves the exact pipeline string.
        assert regions[0].text_raw == 'Abc-1234'

    @pytest.mark.asyncio
    async def test_pick_best_plate_region(self) -> None:
        from src.services.detection.cascade_detect.ocr_recognizer import PaddleOcrTextRecognizer

        pool = MagicMock()
        pool.infer = AsyncMock(
            return_value=_ocr_pipeline_mock_result(
                n=2,
                boxes=[
                    [0.10, 0.40, 0.30, 0.50],  # plate-shaped, small
                    [0.10, 0.40, 0.50, 0.50],  # plate-shaped, larger
                ],
                texts=['XY-12', 'ABC-1234'],
                det_scores=[0.90, 0.90],
                rec_scores=[0.99, 0.80],
            )
        )
        rec = PaddleOcrTextRecognizer(pool, NEUTRAL_REGION_PROFILE)
        regions = await rec.detect_regions(_jpeg_bytes())
        pick = rec.pick_best_text_region(regions)
        # Larger plate-shaped region wins despite lower rec_score.
        assert pick is not None
        assert pick.text == 'ABC-1234'

    @pytest.mark.asyncio
    async def test_rejects_letters_only_text(self) -> None:
        """With the letters+digits rule on, lettering-only text must not promote."""
        from src.services.detection.cascade_detect.ocr_recognizer import PaddleOcrTextRecognizer

        pool = MagicMock()
        pool.infer = AsyncMock(
            return_value=_ocr_pipeline_mock_result(
                n=3,
                boxes=[
                    [0.10, 0.40, 0.50, 0.50],  # plate-shaped, all-letters
                    [0.10, 0.55, 0.50, 0.65],  # plate-shaped, all-letters
                    [0.10, 0.70, 0.45, 0.78],  # plate-shaped, all-letters
                ],
                texts=['FORD', 'DEALER', 'TURBO'],
                det_scores=[0.95, 0.95, 0.95],
                rec_scores=[0.98, 0.95, 0.92],
            )
        )
        rec = PaddleOcrTextRecognizer(pool, LETTERS_AND_DIGITS_PROFILE)
        regions = await rec.detect_regions(_jpeg_bytes())
        # Loose plate-shape filter accepts them, but the stricter
        # is_region_text_candidate rejects all three because they lack digits.
        assert all(not r.is_region_text_candidate for r in regions)
        assert rec.pick_best_text_region(regions) is None

    @pytest.mark.asyncio
    async def test_rejects_low_rec_score(self) -> None:
        """Even a perfect-looking plate text below the rec-score floor is rejected."""
        from src.services.detection.cascade_detect.ocr_recognizer import PaddleOcrTextRecognizer

        pool = MagicMock()
        pool.infer = AsyncMock(
            return_value=_ocr_pipeline_mock_result(
                n=1,
                boxes=[[0.10, 0.40, 0.50, 0.50]],
                texts=['ABC1234'],
                det_scores=[0.95],
                rec_scores=[0.55],  # below profile.text_hint_rec_floor=0.70
            )
        )
        rec = PaddleOcrTextRecognizer(pool, NEUTRAL_REGION_PROFILE)
        regions = await rec.detect_regions(_jpeg_bytes())
        assert len(regions) == 1
        assert not regions[0].is_region_text_candidate
        assert rec.pick_best_text_region(regions) is None

    @pytest.mark.asyncio
    async def test_accepts_vanity_plate(self) -> None:
        """Vanity plates with letter+digit mix promote (e.g. 'LUV2DRV')."""
        from src.services.detection.cascade_detect.ocr_recognizer import PaddleOcrTextRecognizer

        pool = MagicMock()
        pool.infer = AsyncMock(
            return_value=_ocr_pipeline_mock_result(
                n=1,
                boxes=[[0.10, 0.40, 0.45, 0.50]],  # ar=3.5
                texts=['LUV2DRV'],
                det_scores=[0.92],
                rec_scores=[0.88],
            )
        )
        rec = PaddleOcrTextRecognizer(pool, NEUTRAL_REGION_PROFILE)
        regions = await rec.detect_regions(_jpeg_bytes())
        assert regions[0].is_region_text_candidate
        assert rec.pick_best_text_region(regions) is regions[0]

    @pytest.mark.asyncio
    async def test_rejects_non_plate_shaped_region(self) -> None:
        from src.services.detection.cascade_detect.ocr_recognizer import PaddleOcrTextRecognizer

        pool = MagicMock()
        pool.infer = AsyncMock(
            return_value=_ocr_pipeline_mock_result(
                n=1,
                # Aspect 1.0 — square, not plate-shaped.
                boxes=[[0.10, 0.40, 0.30, 0.60]],
                texts=['ABC-1234'],
                det_scores=[0.95],
                rec_scores=[0.95],
            )
        )
        rec = PaddleOcrTextRecognizer(pool, NEUTRAL_REGION_PROFILE)
        regions = await rec.detect_regions(_jpeg_bytes())
        assert len(regions) == 1
        assert rec.pick_best_text_region(regions) is None

    @pytest.mark.asyncio
    async def test_read_plate_region_joins_lines(self) -> None:
        from src.services.detection.cascade_detect.ocr_recognizer import PaddleOcrTextRecognizer

        pool = MagicMock()
        pool.infer = AsyncMock(
            return_value=_ocr_pipeline_mock_result(
                n=2,
                boxes=[
                    [0.10, 0.40, 0.80, 0.55],  # bottom line
                    [0.10, 0.10, 0.80, 0.30],  # top line
                ],
                texts=['1234', 'MOTO'],
                det_scores=[0.95, 0.95],
                rec_scores=[0.95, 0.90],
            )
        )
        rec = PaddleOcrTextRecognizer(pool, NEUTRAL_REGION_PROFILE)
        result = await rec.read_region_text(_jpeg_bytes())
        assert result is not None
        text, conf = result
        # Top-then-bottom reading order: MOTO before 1234.
        assert text == 'MOTO 1234'
        assert conf == pytest.approx(0.925, abs=1e-3)

    @pytest.mark.asyncio
    async def test_infer_failure_returns_empty(self) -> None:
        from src.services.detection.cascade_detect.ocr_recognizer import PaddleOcrTextRecognizer

        pool = MagicMock()
        pool.infer = AsyncMock(side_effect=RuntimeError('triton down'))
        rec = PaddleOcrTextRecognizer(pool, NEUTRAL_REGION_PROFILE)
        regions = await rec.detect_regions(_jpeg_bytes())
        assert regions == []
