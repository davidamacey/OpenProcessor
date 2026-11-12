"""The harness Triton serves the core inference models over the real KServe v2 gRPC wire.

The fake is started in-process and called with ``tritonclient`` (the client library the API uses),
with the API's own pre- and post-processing around it, so a pass proves the fake's tensor names,
dtypes and shapes against the client, not against a copy of them. Protocol:
https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md and
https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_model_repository.html
"""

from __future__ import annotations

import importlib.util
import json
import threading
import urllib.request
from concurrent import futures
from http.server import ThreadingHTTPServer
from pathlib import Path

import grpc
import numpy as np
import pytest
import tritonclient.grpc as triton
from tritonclient.grpc import service_pb2_grpc

from src.utils.face_align import align_faces_batch, preprocess_for_arcface
from src.utils.scrfd_decode import ALL_OUTPUT_NAMES, decode_scrfd_outputs, preprocess_scrfd


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load('fake_triton_core', ROOT / 'docker' / 'test' / 'fake_triton.py')
    monkeypatch.setattr(module, 'MODEL_REPO', tmp_path)
    module._loaded_from_repo.clear()
    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2), options=module.GRPC_OPTIONS
    )
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(module._Models(), grpc_server)
    grpc_port = grpc_server.add_insecure_port('127.0.0.1:0')
    grpc_server.start()
    http = ThreadingHTTPServer(('127.0.0.1', 0), module._Http)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    client = triton.InferenceServerClient(
        f'127.0.0.1:{grpc_port}', channel_args=module.GRPC_OPTIONS
    )
    yield module, client, f'http://127.0.0.1:{http.server_address[1]}'
    client.close()
    grpc_server.stop(0)
    http.shutdown()
    http.server_close()


def _infer(client, model: str, name: str, array: np.ndarray, datatype: str, outputs: list[str]):
    inp = triton.InferInput(name, list(array.shape), datatype)
    inp.set_data_from_numpy(array)
    return client.infer(model, [inp], outputs=[triton.InferRequestedOutput(o) for o in outputs])


def _picture(colour: tuple[int, int, int]) -> np.ndarray:
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    img[100:380, 150:500] = colour
    return img


def test_scrfd_outputs_decode_to_one_face_with_five_landmarks(fake) -> None:
    _module, client, _http = fake
    blob, scale = preprocess_scrfd(_picture((200, 30, 30)))
    result = _infer(client, 'scrfd_10g_bnkps', 'input.1', blob, 'FP32', ALL_OUTPUT_NAMES)
    raw = {name: result.as_numpy(name) for name in ALL_OUTPUT_NAMES}
    assert raw['score_8'].shape == (1, 12800, 1)
    assert raw['kps_32'].shape == (1, 800, 10)
    boxes, scores, landmarks = decode_scrfd_outputs(raw, scale)
    assert boxes.shape == (1, 4)
    assert scores[0] == pytest.approx(0.95)
    assert landmarks.shape == (1, 5, 2)
    x1, y1, x2, y2 = boxes[0] * scale
    assert (x2 - x1, y2 - y1) == (pytest.approx(80.0), pytest.approx(80.0))
    points = landmarks[0] * scale
    assert np.all(points >= [x1, y1])
    assert np.all(points <= [x2, y2 + 24])


def test_arcface_embeds_the_aligned_face_to_a_stable_unit_vector(fake) -> None:
    _module, client, _http = fake
    img = _picture((30, 200, 30))
    blob, scale = preprocess_scrfd(img)
    result = _infer(client, 'scrfd_10g_bnkps', 'input.1', blob, 'FP32', ALL_OUTPUT_NAMES)
    _boxes, _scores, landmarks = decode_scrfd_outputs(
        {n: result.as_numpy(n) for n in ALL_OUTPUT_NAMES}, scale
    )
    crops = preprocess_for_arcface(align_faces_batch(img, landmarks))
    assert crops.shape == (1, 3, 112, 112)
    first = _infer(client, 'arcface_w600k_r50', 'input', crops, 'FP32', ['output']).as_numpy(
        'output'
    )
    again = _infer(client, 'arcface_w600k_r50', 'input', crops, 'FP32', ['output']).as_numpy(
        'output'
    )
    other = _infer(client, 'arcface_w600k_r50', 'input', crops + 1, 'FP32', ['output']).as_numpy(
        'output'
    )
    assert first.shape == (1, 512)
    assert np.linalg.norm(first[0]) == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_array_equal(first, again)
    assert float(first[0] @ other[0]) < 0.5


def test_mobileclip_image_vector_is_nearest_to_its_own_picture(fake) -> None:
    _module, client, _http = fake
    red, blue = (np.zeros((1, 3, 256, 256), np.float32) for _ in range(2))
    red[:, 0] = 2.0
    blue[:, 2] = 2.0

    def vec(array: np.ndarray) -> np.ndarray:
        result = _infer(
            client, 'mobileclip2_s2_image_encoder', 'images', array, 'FP32', ['image_embeddings']
        )
        return result.as_numpy('image_embeddings')[0]

    assert vec(red).shape == (512,)
    assert float(vec(red) @ vec(red)) == pytest.approx(1.0, abs=1e-5)
    assert float(vec(red) @ vec(blue)) < 0.5


def test_mobileclip_text_vector_is_a_function_of_the_tokens(fake) -> None:
    _module, client, _http = fake
    tokens = np.zeros((2, 77), dtype=np.int64)
    tokens[0, :3] = [49406, 320, 49407]
    tokens[1, :3] = [49406, 321, 49407]
    out = _infer(
        client, 'mobileclip2_s2_text_encoder', 'text_tokens', tokens, 'INT64', ['text_embeddings']
    ).as_numpy('text_embeddings')
    assert out.shape == (2, 512)
    assert float(out[0] @ out[1]) < 0.5


def test_model_config_over_grpc_and_http_matches_the_repository_config(fake) -> None:
    _module, client, http = fake
    for name, dims in (
        ('scrfd_10g_bnkps', [3, 640, 640]),
        ('arcface_w600k_r50', [3, 112, 112]),
        ('mobileclip2_s2_image_encoder', [3, 256, 256]),
        ('mobileclip2_s2_text_encoder', [77]),
    ):
        pbtxt = (ROOT / 'models' / name / 'config.pbtxt').read_text()
        config = client.get_model_config(name, as_json=True)['config']
        assert list(map(int, config['input'][0]['dims'])) == dims
        assert f'max_batch_size: {config["max_batch_size"]}' in pbtxt
        with urllib.request.urlopen(f'{http}/v2/models/{name}/config', timeout=5) as r:  # nosec B310
            assert json.loads(r.read())['input'][0]['dims'] == dims


def test_a_promoted_model_is_listed_after_load_and_answers_the_warm_up(fake) -> None:
    module, client, http = fake
    (module.MODEL_REPO / 'promoted_x' / '1').mkdir(parents=True)
    (module.MODEL_REPO / 'promoted_x' / '1' / 'model.onnx').write_bytes(b'x')

    def post(path: str, body: bytes = b''):
        return urllib.request.urlopen(  # nosec B310 - loopback
            urllib.request.Request(f'{http}{path}', data=body, method='POST'), timeout=5
        )

    def listed() -> set[str]:
        with post('/v2/repository/index') as r:
            return {row['name'] for row in json.loads(r.read())}

    assert 'promoted_x' not in listed()
    with post('/v2/repository/models/promoted_x/load') as r:
        assert r.status == 200
    assert 'promoted_x' in listed()
    body = json.dumps(
        {
            'inputs': [
                {'name': 'images', 'shape': [2, 3, 640, 640], 'datatype': 'UINT8', 'data': [0]}
            ]
        }
    ).encode()
    with post('/v2/models/promoted_x/infer', body) as r:
        out = json.loads(r.read())['outputs'][0]
    assert (out['name'], out['shape']) == ('output0', [2, 300, 6])
    grpc_out = _infer(
        client, 'promoted_x', 'images', np.zeros((1, 3, 640, 640), np.uint8), 'UINT8', ['output0']
    ).as_numpy('output0')
    assert grpc_out.shape == (1, 300, 6)
    with post('/v2/repository/models/promoted_x/unload'):
        pass
    assert 'promoted_x' not in listed()


def test_loading_a_model_that_is_not_on_disk_is_refused(fake) -> None:
    _module, _client, http = fake
    request = urllib.request.Request(
        f'{http}/v2/repository/models/ghost/load', data=b'', method='POST'
    )
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(request, timeout=5)  # nosec B310
    assert refused.value.code == 400
