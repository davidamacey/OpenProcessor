"""``/detect`` letterboxes to the size the model was exported at, read from
Triton's model metadata: a model promoted at 320 used to get a 640 tensor and
a shape error."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from src.clients import triton_client as tc


if TYPE_CHECKING:
    import pytest


class _Adapter:
    requested_outputs = ('o',)

    def parse(self, _response: Any, batch_size: int) -> list[dict[str, Any]]:
        return [{'num_dets': 0} for _ in range(batch_size)]


def _client(
    size: int, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, list[tuple[int, ...]], list[str]]:
    sent: list[tuple[int, ...]] = []
    meta_calls: list[str] = []
    meta = SimpleNamespace(inputs=[SimpleNamespace(name='images', shape=[-1, 3, size, size])])

    def get_metadata(name: str) -> Any:
        meta_calls.append(name)
        return meta

    client = tc.TritonClient.__new__(tc.TritonClient)
    client.input_size = 640
    client._input_sizes = {}
    client._detection_adapters = {'m': _Adapter()}  # type: ignore[dict-item]
    client.client = SimpleNamespace(get_model_metadata=get_metadata)

    def infer(_name: str, inputs: list[Any], _outputs: list[Any]) -> None:
        sent.append(tuple(inputs[0].shape()))

    monkeypatch.setattr(client, '_infer_with_retry', infer)
    monkeypatch.setattr(tc, 'InferRequestedOutput', lambda name: name)
    return client, sent, meta_calls


def test_single_and_batch_use_the_model_input_size(monkeypatch: pytest.MonkeyPatch) -> None:
    client, sent, meta_calls = _client(320, monkeypatch)
    img = np.zeros((480, 600, 3), dtype=np.uint8)
    result = client.infer_yolo_end2end(img, 'm')
    client.infer_yolo_end2end_batch([img, img], 'm')
    assert sent == [(1, 3, 320, 320), (2, 3, 320, 320)]
    assert result['input_size'] == 320
    assert meta_calls == ['m']  # cached after the first read
