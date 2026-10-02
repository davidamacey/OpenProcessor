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
    shape = [-1, 3, size, size]

    def get_metadata(name: str) -> Any:
        meta_calls.append(name)
        return SimpleNamespace(
            inputs=[SimpleNamespace(name='images', shape=shape)],
            outputs=[SimpleNamespace(name='n', shape=[-1, 1]) for _ in range(1)],
        )

    client = tc.TritonClient.__new__(tc.TritonClient)
    client.input_size = 640
    client._model_info = {}
    monkeypatch.setattr(tc, 'resolve_adapter', lambda _meta: _Adapter())
    client.client = SimpleNamespace(get_model_metadata=get_metadata)

    def infer(_name: str, inputs: list[Any], _outputs: list[Any]) -> None:
        sent.append(tuple(inputs[0].shape()))

    monkeypatch.setattr(client, '_infer_with_retry', infer)
    monkeypatch.setattr(tc, 'InferRequestedOutput', lambda name: name)
    client.reshape = lambda new: shape.__setitem__(slice(None), [-1, 3, new, new])  # type: ignore[attr-defined]
    return client, sent, meta_calls


def test_single_and_batch_use_the_model_input_size(monkeypatch: pytest.MonkeyPatch) -> None:
    client, sent, meta_calls = _client(320, monkeypatch)
    img = np.zeros((480, 600, 3), dtype=np.uint8)
    result = client.infer_yolo_end2end(img, 'm')
    client.infer_yolo_end2end_batch([img, img], 'm')
    assert sent == [(1, 3, 320, 320), (2, 3, 320, 320)]
    assert result['input_size'] == 320
    assert meta_calls == ['m']  # cached after the first read


def test_a_model_replaced_at_another_size_is_picked_up_after_the_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review finding: the first size read was cached forever, so a model
    re-promoted under the same name kept getting the old tensor shape."""
    now = [1000.0]
    monkeypatch.setattr(tc.time, 'monotonic', lambda: now[0])
    client, _sent, meta_calls = _client(640, monkeypatch)
    assert client._model_input_size('m') == 640
    client.reshape(1280)
    assert client._model_input_size('m') == 640  # still inside the TTL
    now[0] += tc._MODEL_INFO_TTL_S + 1
    assert client._model_input_size('m') == 1280
    assert meta_calls == ['m', 'm']


def test_the_ingest_clip_path_letterboxes_to_the_detector_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, sent, _ = _client(320, monkeypatch)
    monkeypatch.setattr(tc.TritonModelConfig, 'YOLO_MODEL', 'm')
    monkeypatch.setattr(
        client, '_preprocess_clip_cpu', lambda _img: np.zeros((1, 3, 256, 256), dtype=np.float32)
    )
    img = np.zeros((480, 600, 3), dtype=np.uint8)
    buf = tc.io.BytesIO()
    tc.Image.fromarray(img).save(buf, format='JPEG')

    class _Resp:
        def as_numpy(self, _n: str) -> np.ndarray:
            return np.zeros((1, 512), dtype=np.float32)

    def infer(_name: str, inputs: list[Any], _outputs: list[Any]) -> Any:
        sent.append(tuple(inputs[0].shape()))
        return _Resp()

    monkeypatch.setattr(client, '_infer_with_retry', infer)
    result = client.infer_yolo_clip_cpu(buf.getvalue())
    assert sent[0] == (1, 3, 320, 320)
    assert result['input_size'] == 320
