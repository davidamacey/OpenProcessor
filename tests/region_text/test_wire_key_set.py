"""PT-F4: the box wire carries the readable text fields; the match keys stay in storage."""

from __future__ import annotations

from src.domain.region_boxes import RegionBox
from src.services.curation.wire import region_box_to_wire


def test_the_box_wire_key_set_is_pinned() -> None:
    box = RegionBox(
        box_id='b1',
        bbox_norm=(0.1, 0.1, 0.5, 0.5),
        state='accepted',
        text='abo-1i2-xyz-987',
        text_frame='abo-1i2-xyz-987',
        text_norm='ABO1I2XYZ987',
        text_fold='AB0112XYZ987',
        text_reader='paddleocr_rec_trt',
        text_reader_version='1',
    )

    wire = region_box_to_wire({}, box)

    assert wire['text'] == 'abo-1i2-xyz-987'
    assert wire['text_frame'] == 'abo-1i2-xyz-987'
    assert (wire['text_reader'], wire['text_reader_version']) == ('paddleocr_rec_trt', '1')
    assert 'text_conf_char_min' in wire
    assert 'text_norm' not in wire
    assert 'text_fold' not in wire
    assert 'text_engine_version' not in wire
    assert sorted(k for k in wire if k.startswith('text')) == [
        'text',
        'text_choice',
        'text_conf_char_min',
        'text_confidence',
        'text_disagreement',
        'text_frame',
        'text_ocr',
        'text_raw',
        'text_reader',
        'text_reader_version',
        'text_source',
        'text_vlm',
        'text_vlm_invalid',
    ]
