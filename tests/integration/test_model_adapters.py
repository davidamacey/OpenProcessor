"""Adapter-registry behavior pins for dual-family YOLO serving.

The registry must pick the right parser purely from Triton model metadata
(output names/shapes), so YOLO11 end2end (EfficientNMS 4-tensor) and
YOLO26 (fused single-tensor) engines serve side by side without name
conventions.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.clients.model_adapters import End2EndNMSAdapter, FusedDetAdapter, resolve_adapter


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


def test_resolve_end2end_signature() -> None:
    assert isinstance(resolve_adapter(_end2end_metadata()), End2EndNMSAdapter)


@pytest.mark.parametrize('shape', [[-1, 300, 6], [300, 6]])
def test_resolve_fused_signature_with_and_without_batch_axis(shape: list[int]) -> None:
    adapter = resolve_adapter(_Metadata([_Output('output0', shape)]))
    assert isinstance(adapter, FusedDetAdapter)
    assert adapter.output_name == 'output0'


def test_resolve_rejects_unknown_signature() -> None:
    with pytest.raises(ValueError, match='neither'):
        resolve_adapter(_Metadata([_Output('embeddings', [512])]))


def test_end2end_parse_truncates_to_num_dets() -> None:
    batch = 2
    tensors = {
        'num_dets': np.array([[2], [0]], dtype=np.int32),
        'det_boxes': np.zeros((batch, 300, 4), dtype=np.float32),
        'det_scores': np.zeros((batch, 300), dtype=np.float32),
        'det_classes': np.zeros((batch, 300), dtype=np.int32),
    }
    tensors['det_scores'][0, :2] = [0.9, 0.8]
    tensors['det_classes'][0, :2] = [3, 1]

    results = End2EndNMSAdapter().parse(_Response(tensors), batch)

    assert results[0]['num_dets'] == 2
    assert results[0]['boxes'].shape == (2, 4)
    assert list(results[0]['classes']) == [3, 1]
    assert results[1]['num_dets'] == 0
    assert results[1]['boxes'].shape == (0, 4)


def test_fused_parse_drops_zero_score_padding() -> None:
    rows = np.zeros((2, 300, 6), dtype=np.float32)
    rows[0, 0] = [10, 20, 110, 220, 0.9, 3]
    rows[0, 1] = [5, 5, 50, 50, 0.4, 1]
    rows[1, 0] = [1, 2, 3, 4, 0.7, 0]

    results = FusedDetAdapter('output0').parse(_Response({'output0': rows}), 2)

    assert results[0]['num_dets'] == 2
    assert results[1]['num_dets'] == 1
    # Same normalized contract as the end2end adapter
    assert set(results[0]) == {'num_dets', 'boxes', 'scores', 'classes'}
    assert results[0]['classes'].dtype == np.int32
    assert results[0]['boxes'].shape == (2, 4)
    assert pytest.approx(results[0]['scores'][0], abs=1e-6) == 0.9
