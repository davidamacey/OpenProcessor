"""The engine and encoder contracts reject what they must and accept today's inputs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.models import EncoderInputSpec, EngineInputContract, EngineOutputContract


DETECTOR = {
    'dtype': 'fp32',
    'layout': 'NCHW',
    'size': [640, 640],
    'color': 'RGB',
    'normalize': {'mean': [0, 0, 0], 'std': [1, 1, 1], 'scale': 0.00392156862745098},
    'resize': 'letterbox',
    'interpolation': 'bilinear',
    'fill': 114,
    'pad_split': 'ultralytics_round',
}
# src/utils/pe_preprocess.py: shorter edge to 336 with Pillow BILINEAR, centre square, mean/std 0.5.
PE_IMAGE = {
    'dtype': 'uint8',
    'layout': 'NCHW',
    'size': [336, 336],
    'color': 'RGB',
    'normalize': 'in_graph',
    'resize': 'short_edge_center_crop',
    'interpolation': 'pil_bilinear',
}
# A recognizer: fixed height, a dynamic width range, the crop keeps its aspect ratio.
OCR = {
    'dtype': 'fp32',
    'layout': 'NCHW',
    'size': [48, {'min': 32, 'opt': 320, 'max': 960}],
    'color': 'BGR',
    'normalize': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'scale': 0.00392156862745098},
    'resize': 'native_width',
    'interpolation': 'bilinear',
    'max_side': 960,
}


@pytest.mark.parametrize('raw', [DETECTOR, PE_IMAGE, OCR], ids=['detector', 'pe', 'ocr'])
def test_todays_inputs_validate_and_round_trip(raw: dict) -> None:
    contract = EngineInputContract.model_validate(raw)
    assert EngineInputContract.model_validate(contract.model_dump(mode='json')) == contract


def test_an_unknown_resize_value_fails() -> None:
    with pytest.raises(ValidationError, match='resize'):
        EngineInputContract.model_validate({**DETECTOR, 'resize': 'squash'})


@pytest.mark.parametrize(
    'size',
    [
        {'min': 8, 'opt': 4, 'max': 16},
        {'min': 4, 'opt': 32, 'max': 16},
        [640, {'min': 9, 'opt': 5, 'max': 20}],
    ],
)
def test_a_size_range_must_be_ordered(size) -> None:
    with pytest.raises(ValidationError, match='min <= opt <= max'):
        EngineInputContract.model_validate({**DETECTOR, 'size': size})


def test_a_fixed_size_must_be_positive() -> None:
    with pytest.raises(ValidationError, match='at least 1'):
        EngineInputContract.model_validate({**DETECTOR, 'size': [0, 640]})


def test_an_unknown_field_fails() -> None:
    with pytest.raises(ValidationError, match='extra_forbidden'):
        EngineInputContract.model_validate({**DETECTOR, 'stride': 32})


def test_output_contract_needs_one_dtype_per_output() -> None:
    ok = {'kind': 'end2end_boxes', 'shapes': [[-1, 300, 6]], 'dtypes': ['fp32']}
    assert EngineOutputContract.model_validate(ok).kind == 'end2end_boxes'
    with pytest.raises(ValidationError, match='same outputs'):
        EngineOutputContract.model_validate({**ok, 'dtypes': ['fp32', 'int32']})
    with pytest.raises(ValidationError, match='kind'):
        EngineOutputContract.model_validate({**ok, 'kind': 'masks'})


def test_encoder_spec_embeds_the_engine_image_contract() -> None:
    spec = EncoderInputSpec.model_validate(
        {
            'image': PE_IMAGE,
            'text': {'context_length': 32, 'prefixes': {'query': 'a photo of '}},
        }
    )
    assert spec.image == EngineInputContract.model_validate(PE_IMAGE)
    assert spec.text is not None
    assert spec.text.prefixes.document == ''


def test_encoder_spec_needs_a_modality_and_forbids_extras() -> None:
    with pytest.raises(ValidationError, match='image or a text'):
        EncoderInputSpec.model_validate({})
    with pytest.raises(ValidationError, match='extra_forbidden'):
        EncoderInputSpec.model_validate({'image': PE_IMAGE, 'audio': {}})
