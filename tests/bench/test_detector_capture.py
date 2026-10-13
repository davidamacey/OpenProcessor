"""``detector_capture.gpu_decode_row``: the GPU decode's rows mapped like the CPU capture's."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scripts.bench import detector_capture as dc


class _Result:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self.outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray:
        return self.outputs[name]


class _Client:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self.outputs = outputs
        self.callback: Any = None
        self.stopped = False

    def start_stream(self, callback: Any) -> None:
        self.callback = callback

    def async_stream_infer(self, model: str, inputs: list) -> None:
        assert model == 'gpu_image_prep'
        assert inputs[0].shape() == [4]
        self.callback(result=_Result(self.outputs), error=None)

    def stop_stream(self) -> None:
        self.stopped = True


def test_rows_are_undone_with_the_returned_letterbox() -> None:
    scale, pad = 0.5, (0.0, 80.0)
    client = _Client(
        {
            'status': np.array([b'ok'], dtype=np.object_),
            'letterbox': np.array([scale, *pad]),
            'det_boxes': np.array([[0.25, 0.25, 0.5, 0.5]], np.float32),
            'det_scores': np.array([0.8], np.float32),
            'det_classes': np.array([3], np.int32),
        }
    )
    rows = dc.gpu_decode_row(client, 'gpu_image_prep', b'\xff\xd8\xff\xd9')
    expected = dc.decode_row(
        np.array([1]),
        client.outputs['det_boxes'],
        client.outputs['det_scores'],
        np.array([3]),
        scale,
        pad,
    )
    assert rows == expected
    assert rows[0][:4] == [320.0, 160.0, 640.0, 480.0]
    assert client.stopped


def test_a_declined_image_stops_the_capture() -> None:
    client = _Client({'status': np.array([b'progressive'], dtype=np.object_)})
    with pytest.raises(SystemExit, match='progressive'):
        dc.gpu_decode_row(client, 'gpu_image_prep', b'\xff\xd8\xff\xd9')
