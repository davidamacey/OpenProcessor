"""Region-text OCR reader: dominant-text selection, normalization, reader modes."""

from __future__ import annotations

import pytest
from PIL import Image

from src.config import DetectionProfile
from src.services.detection.cascade_detect import (
    OCR_DET_MAX_SIDE,
    OCR_DET_MIN_SIDE,
    _unframe_line,
    frame_for_ocr,
    ocr_det_input_size,
)
from src.services.detection.reference_profiles import REFERENCE_LICENSE_PLATE_PROFILE
from src.services.detection.region_text import (
    DominantTextConfig,
    OcrLine,
    TextNormalizer,
    ocr_engine_id,
    ocr_needed,
    read_dominant_text,
    resolve_region_text,
    texts_disagree,
    validate_text_reader,
)


REF = DominantTextConfig.from_profile(REFERENCE_LICENSE_PLATE_PROFILE)
GENERIC = DominantTextConfig.from_profile(DetectionProfile(name='generic'))


def _line(text: str, x1: float, y1: float, x2: float, y2: float, score: float = 0.9) -> OcrLine:
    return OcrLine(text=text, box=(x1, y1, x2, y2), score=score)


# A realistic region crop: small header line on top, tall main text in the
# middle, small slogan below, and frame text hugging the bottom border.
HEADER = _line('Ohio', 0.35, 0.12, 0.65, 0.24, 0.95)
MAIN = _line('ABC-1234', 0.12, 0.30, 0.88, 0.72, 0.91)
SLOGAN = _line('Birthplace of Aviation', 0.20, 0.75, 0.80, 0.85, 0.80)
FRAME = _line('SMITH MOTORS', 0.15, 0.93, 0.85, 0.99, 0.88)


class TestDominantText:
    def test_only_the_main_text_survives(self) -> None:
        r = read_dominant_text([HEADER, MAIN, SLOGAN, FRAME], REF)
        assert r.text == 'ABC1234'
        assert r.kept == (MAIN,)
        assert r.confidence == pytest.approx(0.91)
        assert r.reason == 'ok'

    def test_raw_keeps_every_line_in_reading_order(self) -> None:
        r = read_dominant_text([FRAME, SLOGAN, MAIN, HEADER], REF)
        assert r.raw == 'Ohio ABC-1234 Birthplace of Aviation SMITH MOTORS'

    def test_border_line_is_dropped_even_when_tall(self) -> None:
        tall_edge = _line('WWWWWW', 0.0, 0.0, 0.06, 0.9)
        r = read_dominant_text([tall_edge, MAIN], REF)
        assert r.text == 'ABC1234'

    def test_split_main_text_joins_left_to_right(self) -> None:
        right = _line('1234', 0.55, 0.32, 0.88, 0.70, 0.8)
        left = _line('ABC', 0.12, 0.30, 0.45, 0.72, 0.9)
        r = read_dominant_text([right, SLOGAN, left], REF)
        assert r.text == 'ABC1234'
        assert r.confidence == pytest.approx(0.8)
        assert r.confidence_mean == pytest.approx(0.85)

    def test_two_equal_rows_read_top_then_bottom(self) -> None:
        top = _line('AB', 0.2, 0.15, 0.8, 0.45)
        bottom = _line('123', 0.2, 0.52, 0.8, 0.82)
        r = read_dominant_text([bottom, top], REF)
        assert r.text == 'AB123'

    def test_large_issuer_line_is_a_stopword_not_the_reference_height(self) -> None:
        big_header = _line('NEW YORK', 0.1, 0.10, 0.9, 0.45)
        main = _line('XYZ 987', 0.15, 0.50, 0.85, 0.80)
        r = read_dominant_text([big_header, main], REF)
        assert r.text == 'XYZ987'

    def test_single_word_stopword_is_removed_inside_a_line(self) -> None:
        r = read_dominant_text([_line('TEXAS 4GH 221', 0.1, 0.3, 0.9, 0.7)], REF)
        assert r.text == '4GH221'

    def test_stopwords_ignored_without_a_list(self) -> None:
        r = read_dominant_text([_line('TEXAS', 0.1, 0.3, 0.9, 0.7)], GENERIC)
        assert r.text == 'TEXAS'

    def test_length_bounds(self) -> None:
        assert read_dominant_text([_line('A', 0.3, 0.3, 0.6, 0.7)], REF).reason == 'too_short'
        long_line = _line('ABCDEFGHIJK', 0.1, 0.3, 0.9, 0.7)
        r = read_dominant_text([long_line], REF)
        assert r.text is None
        assert r.reason == 'too_long'
        assert r.raw == 'ABCDEFGHIJK'

    def test_min_confidence(self) -> None:
        cfg = DominantTextConfig(normalizer=REF.normalizer, join='', min_confidence=0.6)
        r = read_dominant_text([_line('ABC123', 0.1, 0.3, 0.9, 0.7, 0.4)], cfg)
        assert r.text is None
        assert r.reason == 'low_confidence'

    def test_no_lines(self) -> None:
        r = read_dominant_text([], REF)
        assert (r.text, r.raw, r.reason) == (None, '', 'no_lines')

    def test_generic_defaults_keep_case_and_spaces_between_pieces(self) -> None:
        r = read_dominant_text([_line('Lot 42', 0.1, 0.3, 0.9, 0.7)], GENERIC)
        assert r.text == 'Lot42'
        pieces = [_line('Lot', 0.1, 0.3, 0.4, 0.7), _line('42', 0.5, 0.3, 0.9, 0.7)]
        assert read_dominant_text(pieces, GENERIC).text == 'Lot 42'


class TestNormalizer:
    def test_reference_charset(self) -> None:
        assert REF.normalizer.normalize('ab-12 Ü') == 'AB12'

    def test_empty_charset_keeps_non_space(self) -> None:
        assert TextNormalizer().normalize(' a b-c ') == 'ab-c'


class TestReaderModes:
    def _ocr(self, text: str | None = 'ABC1234'):
        lines = [MAIN] if text else []
        return read_dominant_text(lines, REF)

    def test_validate(self) -> None:
        assert validate_text_reader('both') == 'both'
        with pytest.raises(ValueError, match='unknown text_reader'):
            validate_text_reader('best')

    @pytest.mark.parametrize(
        ('mode', 'vlm_text', 'vlm_available', 'expected'),
        [
            ('vlm', 'X', True, False),
            ('vlm', None, True, False),
            ('vlm', None, False, True),
            ('ocr', 'X', True, True),
            ('both', 'X', True, True),
            ('vlm_then_ocr', 'X', True, False),
            ('vlm_then_ocr', None, True, True),
        ],
    )
    def test_ocr_needed(self, mode, vlm_text, vlm_available, expected) -> None:
        assert ocr_needed(mode, vlm_text=vlm_text, vlm_available=vlm_available) is expected

    def _resolve(self, mode: str, vlm_text: str | None, ocr):
        return resolve_region_text(
            mode,
            vlm_text=vlm_text,
            vlm_confidence='high',
            vlm_engine='vlm-model',
            ocr=ocr,
            ocr_engine='det:1+rec:1',
            normalizer=REF.normalizer,
        )

    def test_both_agree_after_normalization(self) -> None:
        out = self._resolve('both', 'abc 1234', self._ocr())
        assert out['text'] == 'abc 1234'
        assert out['text_source'] == 'vlm'
        assert out['text_vlm'] == 'abc 1234'
        assert out['text_ocr'] == 'ABC1234'
        assert out['text_disagreement'] is False
        assert out['text_raw'] == 'ABC-1234'

    def test_both_disagree(self) -> None:
        out = self._resolve('both', 'ABC1284', self._ocr())
        assert out['text_disagreement'] is True
        assert out['text'] == 'ABC1284'

    def test_both_without_vlm_text_uses_ocr_and_no_flag(self) -> None:
        out = self._resolve('both', None, self._ocr())
        assert out['text'] == 'ABC1234'
        assert out['text_source'] == 'ocr'
        assert out['text_engine_version'] == 'det:1+rec:1'
        assert out['text_confidence'] == pytest.approx(0.91)
        assert 'text_disagreement' not in out
        assert 'text_vlm' not in out

    def test_ocr_mode_prefers_ocr(self) -> None:
        out = self._resolve('ocr', 'ZZZ', self._ocr())
        assert (out['text'], out['text_source']) == ('ABC1234', 'ocr')

    def test_vlm_mode_keeps_vlm_text(self) -> None:
        out = self._resolve('vlm', 'ZZZ', None)
        assert out == {
            'text_vlm': 'ZZZ',
            'text': 'ZZZ',
            'text_source': 'vlm',
            'text_engine_version': 'vlm-model',
            'text_confidence': 0.92,
            'text_raw': 'ZZZ',
            'text_choice': 'vlm_only',
        }

    def test_nothing_read_writes_nothing(self) -> None:
        assert self._resolve('both', None, self._ocr(None)) == {}

    def test_rejected_ocr_reading_still_records_raw(self) -> None:
        reading = read_dominant_text([_line('ABCDEFGHIJKL', 0.1, 0.3, 0.9, 0.7)], REF)
        assert self._resolve('ocr', None, reading) == {'text_raw': 'ABCDEFGHIJKL'}

    def test_disagree_needs_both(self) -> None:
        assert texts_disagree(None, 'A', REF.normalizer) is None
        assert texts_disagree('--', 'A', REF.normalizer) is None

    def test_engine_id(self) -> None:
        assert (
            ocr_engine_id(REFERENCE_LICENSE_PLATE_PROFILE)
            == 'paddleocr_det_trt:1+paddleocr_rec_trt:1'
        )


class TestOcrFraming:
    @pytest.mark.parametrize(
        ('w', 'h'), [(39, 25), (640, 256), (4000, 100), (50, 3000), (960, 960)]
    )
    def test_det_input_size_is_inside_the_engine_window(self, w: int, h: int) -> None:
        dw, dh = ocr_det_input_size(w, h)
        for side in (dw, dh):
            assert OCR_DET_MIN_SIDE <= side <= OCR_DET_MAX_SIDE
            assert side % 32 == 0

    @pytest.mark.parametrize(('w', 'h'), [(39, 25), (12, 7), (300, 120), (2000, 60)])
    def test_frame_upscales_small_crops_and_round_trips_boxes(self, w: int, h: int) -> None:
        img = Image.new('RGB', (w, h), (200, 200, 200))
        canvas, (ox, oy, sw, sh) = frame_for_ocr(img, min_height=56)
        cw, ch = canvas.size
        assert OCR_DET_MIN_SIDE <= cw <= OCR_DET_MAX_SIDE
        assert OCR_DET_MIN_SIDE <= ch <= OCR_DET_MAX_SIDE
        assert cw % 32 == 0
        assert ch % 32 == 0
        assert ox + sw <= cw
        assert oy + sh <= ch
        if h < 56 and w * 56 / h <= OCR_DET_MAX_SIDE - 64:
            assert sh == 56
        # A box covering the pasted crop maps back to the whole crop.
        line = OcrLine('X', (ox / cw, oy / ch, (ox + sw) / cw, (oy + sh) / ch), 0.9)
        back = _unframe_line(line, (cw, ch), (ox, oy, sw, sh))
        assert back is not None
        assert back.box == pytest.approx((0.0, 0.0, 1.0, 1.0))

    def test_line_entirely_in_the_padding_is_dropped(self) -> None:
        img = Image.new('RGB', (40, 20))
        canvas, placement = frame_for_ocr(img, min_height=56)
        line = OcrLine('X', (0.0, 0.0, 0.01, 0.01), 0.9)
        assert _unframe_line(line, canvas.size, placement) is None
