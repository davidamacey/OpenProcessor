"""Adapter-registry behavior pins for the one fused detector contract.

The registry picks the parser purely from Triton model metadata (output names/shapes): every
family serves the single ``[..., 300, 6]`` tensor, and anything else fails closed with the
export command.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.clients.model_adapters import FusedDetAdapter, resolve_adapter


pytestmark = pytest.mark.integration


class _Output:
    def __init__(self, name: str, shape: list[int]):
        self.name = name
        self.shape = shape


class _Metadata:
    def __init__(self, outputs: list[_Output]):
        self.outputs = outputs


class _Response:
    """Duck-typed Triton response backed by a dict of arrays."""

    def __init__(self, tensors: dict[str, np.ndarray]):
        self._tensors = tensors

    def as_numpy(self, name: str) -> np.ndarray:
        return self._tensors[name]


def _end2end_metadata() -> _Metadata:
    return _Metadata(
        [
            _Output('num_dets', [1]),
            _Output('det_boxes', [300, 4]),
            _Output('det_scores', [300]),
            _Output('det_classes', [300]),
        ]
    )


def test_resolve_refuses_the_four_tensor_signature_naming_the_export_command() -> None:
    with pytest.raises(ValueError, match='unsupported_detector_contract') as err:
        resolve_adapter(_end2end_metadata())
    assert 'export' in str(err.value)


@pytest.mark.parametrize('shape', [[-1, 300, 6], [300, 6]])
def test_resolve_fused_signature_with_and_without_batch_axis(shape: list[int]) -> None:
    adapter = resolve_adapter(_Metadata([_Output('output0', shape)]))
    assert isinstance(adapter, FusedDetAdapter)
    assert adapter.output_name == 'output0'


def test_resolve_rejects_unknown_signature() -> None:
    with pytest.raises(ValueError, match='unsupported_detector_contract'):
        resolve_adapter(_Metadata([_Output('embeddings', [512])]))


@pytest.mark.parametrize('shape', [[-1, 100, 6], [-1, 5, 8400], [-1, 300, 7], [6]])
def test_resolve_refuses_a_single_tensor_that_is_not_300_by_6(shape: list[int]) -> None:
    with pytest.raises(ValueError, match='unsupported_detector_contract'):
        resolve_adapter(_Metadata([_Output('output0', shape)]))


def test_fused_parse_drops_zero_score_padding() -> None:
    rows = np.zeros((2, 300, 6), dtype=np.float32)
    rows[0, 0] = [10, 20, 110, 220, 0.9, 3]
    rows[0, 1] = [5, 5, 50, 50, 0.4, 1]
    rows[1, 0] = [1, 2, 3, 4, 0.7, 0]

    results = FusedDetAdapter('output0').parse(_Response({'output0': rows}), 2)

    assert results[0]['num_dets'] == 2
    assert results[1]['num_dets'] == 1
    assert set(results[0]) == {'num_dets', 'boxes', 'scores', 'classes', 'box_frame'}
    assert results[0]['classes'].dtype == np.int32
    assert results[0]['boxes'].shape == (2, 4)
    assert pytest.approx(results[0]['scores'][0], abs=1e-6) == 0.9
