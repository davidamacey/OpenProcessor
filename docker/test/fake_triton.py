"""A Triton stand-in for the live harness: liveness, readiness, a model repository index and four
deterministic models.

The API refuses to start without a live Triton (src/app/startup_gate.py, owner decision D-69:
no degraded mode). The harness verifies the write paths without a GPU, so this answers
``ServerLive`` and ``ServerReady`` over Triton's gRPC protocol, and over HTTP the readiness probe
and the repository index (the config vocabulary lists the served detectors from it).

Four models answer over gRPC so the ingest, search, region-text and OCR routes run end to end
(the fixed outputs are in ``_MODELS`` and the helpers below it): the fused detector (one box per
picture), the PE image encoder (a vector from the picture's coarse colour layout, so a picture is
its own nearest neighbour), the PE text encoder (a vector from the token ids, so it cannot read a
query) and the OCR pipeline (one fixed line). Every other model and call stays unimplemented: a
code path that needs other inference still fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import struct
import threading
from concurrent import futures
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import NamedTuple

import grpc
import numpy as np
from tritonclient.grpc import model_config_pb2, service_pb2, service_pb2_grpc
from tritonclient.utils import np_to_triton_dtype, triton_to_np_dtype


#: The text the OCR stand-in reads on every image, and its score.
FAKE_OCR_TEXT = 'FAKE-PLATE-0042'
EMBED_DIM = 1024
MAX_DET = 300


_CONFIG_TYPES = {
    'UINT8': model_config_pb2.TYPE_UINT8,
    'INT32': model_config_pb2.TYPE_INT32,
    'INT64': model_config_pb2.TYPE_INT64,
    'FP32': model_config_pb2.TYPE_FP32,
    'BYTES': model_config_pb2.TYPE_STRING,
}


class _Tensor(NamedTuple):
    name: str
    datatype: str
    shape: list[int]


def _tensor(name: str, datatype: str, shape: list[int]) -> _Tensor:
    return _Tensor(name, datatype, shape)


#: model -> (max_batch_size, inputs, outputs). Shapes follow models/<name>/config.pbtxt.
_MODELS: dict[str, tuple[int, list[_Tensor], list[_Tensor]]] = {
    'yolov11_small_trt_end2end': (
        8,
        [_tensor('images', 'UINT8', [-1, 3, 640, 640])],
        [_tensor('output0', 'FP32', [-1, MAX_DET, 6])],
    ),
    'pe_image_encoder': (
        32,
        [_tensor('images', 'UINT8', [-1, 3, 336, 336])],
        [_tensor('image_embeddings', 'FP32', [-1, EMBED_DIM])],
    ),
    'pe_text_encoder': (
        8,
        [_tensor('text_tokens', 'INT64', [-1, -1])],
        [_tensor('text_embeddings', 'FP32', [-1, EMBED_DIM])],
    ),
    'ocr_pipeline': (
        0,
        [
            _tensor('ocr_images', 'UINT8', [3, -1, -1]),
            _tensor('original_image', 'UINT8', [3, -1, -1]),
            _tensor('orig_shape', 'INT32', [2]),
        ],
        [
            _tensor('num_texts', 'INT32', [1]),
            _tensor('text_boxes', 'FP32', [128, 8]),
            _tensor('text_boxes_normalized', 'FP32', [128, 4]),
            _tensor('texts', 'BYTES', [128]),
            _tensor('text_scores', 'FP32', [128]),
            _tensor('rec_scores', 'FP32', [128]),
            _tensor('rec_char_min', 'FP32', [128]),
        ],
    ),
}


#: Models the repository lists although this stand-in never infers on them: the OCR pipeline's own
#: detection and recognition members (a profile that names the pipeline checks the recogniser).
_LISTED_ONLY = ('paddleocr_det_trt', 'paddleocr_rec_trt')


def _repository_index() -> list[dict[str, str]]:
    return [
        {'name': name, 'version': '1', 'state': 'READY', 'reason': ''}
        for name in (*_MODELS, *_LISTED_ONLY)
    ]


def _model_of(path: str, prefix: str, suffix: str) -> str:
    return path.removeprefix(prefix).removesuffix(suffix) if path.startswith(prefix) else ''


def _config_json(name: str) -> dict[str, object]:
    """Triton's HTTP model-configuration document (the fields the API reads)."""
    max_batch, inputs, outputs = _MODELS[name]

    def tensors(specs: list[_Tensor]) -> list[dict[str, object]]:
        return [
            {
                'name': t.name,
                'data_type': f'TYPE_{"STRING" if t.datatype == "BYTES" else t.datatype}',
                'dims': t.shape[1:] if max_batch else t.shape,
            }
            for t in specs
        ]

    return {
        'name': name,
        'max_batch_size': max_batch,
        'input': tensors(inputs),
        'output': tensors(outputs),
    }


def _detect(images: np.ndarray) -> np.ndarray:
    """One box over the middle of every letterboxed frame, class 0, score 0.9."""
    size = float(images.shape[-1])
    out = np.zeros((images.shape[0], MAX_DET, 6), dtype=np.float32)
    out[:, 0, :] = [0.2 * size, 0.2 * size, 0.8 * size, 0.8 * size, 0.9, 0.0]
    return out


def _unit(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return (rows / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def _image_embeddings(images: np.ndarray) -> np.ndarray:
    """An 8x8 grid of mean colours, centred on mid-grey and tiled to 1024 dimensions."""
    n, c, h, w = images.shape
    grid = images.astype(np.float32).reshape(n, c, 8, h // 8, 8, w // 8).mean(axis=(3, 5))
    flat = (grid - 128.0).reshape(n, -1)
    return _unit(np.tile(flat, (1, EMBED_DIM // flat.shape[1] + 1))[:, :EMBED_DIM])


def _text_embeddings(tokens: np.ndarray) -> np.ndarray:
    rows = []
    for row in tokens:
        seed = int.from_bytes(hashlib.sha256(row.tobytes()).digest()[:8], 'little')
        rows.append(np.random.default_rng(seed).standard_normal(EMBED_DIM))
    return _unit(np.asarray(rows, dtype=np.float32))


def _bytes_tensor(values: list[bytes]) -> bytes:
    return b''.join(struct.pack('<I', len(v)) + v for v in values)


def _ocr() -> dict[str, tuple[str, list[int], bytes]]:
    boxes = np.zeros((128, 4), dtype=np.float32)
    boxes[0] = [0.1, 0.4, 0.9, 0.6]
    quads = np.zeros((128, 8), dtype=np.float32)
    quads[0] = [10, 40, 90, 40, 90, 60, 10, 60]
    scores = np.zeros(128, dtype=np.float32)
    scores[0] = 0.97
    texts = [FAKE_OCR_TEXT.encode()] + [b''] * 127
    return {
        'num_texts': ('INT32', [1], np.array([1], dtype=np.int32).tobytes()),
        'text_boxes': ('FP32', [128, 8], quads.tobytes()),
        'text_boxes_normalized': ('FP32', [128, 4], boxes.tobytes()),
        'texts': ('BYTES', [128], _bytes_tensor(texts)),
        'text_scores': ('FP32', [128], scores.tobytes()),
        'rec_scores': ('FP32', [128], scores.tobytes()),
        'rec_char_min': ('FP32', [128], scores.tobytes()),
    }


def _array(request: service_pb2.ModelInferRequest, index: int) -> np.ndarray:
    spec = request.inputs[index]
    dtype = triton_to_np_dtype(spec.datatype)
    return np.frombuffer(request.raw_input_contents[index], dtype=dtype).reshape(list(spec.shape))


def _as_response(
    request: service_pb2.ModelInferRequest, outputs: dict[str, tuple[str, list[int], bytes]]
) -> service_pb2.ModelInferResponse:
    response = service_pb2.ModelInferResponse(
        model_name=request.model_name, model_version=request.model_version, id=request.id
    )
    wanted = [o.name for o in request.outputs] or list(outputs)
    for name in wanted:
        datatype, shape, raw = outputs[name]
        response.outputs.add(name=name, datatype=datatype, shape=shape)
        response.raw_output_contents.append(raw)
    return response


def _fixed(array: np.ndarray) -> tuple[str, list[int], bytes]:
    return (np_to_triton_dtype(array.dtype), list(array.shape), array.tobytes())


class _Models(service_pb2_grpc.GRPCInferenceServiceServicer):
    def ServerLive(self, _request: object, _context: object) -> service_pb2.ServerLiveResponse:
        return service_pb2.ServerLiveResponse(live=True)

    def ServerReady(self, _request: object, _context: object) -> service_pb2.ServerReadyResponse:
        return service_pb2.ServerReadyResponse(ready=True)

    def ModelReady(
        self, request: service_pb2.ModelReadyRequest, _context: object
    ) -> service_pb2.ModelReadyResponse:
        return service_pb2.ModelReadyResponse(ready=request.name in _MODELS)

    def ModelMetadata(
        self, request: service_pb2.ModelMetadataRequest, context: grpc.ServicerContext
    ) -> service_pb2.ModelMetadataResponse:
        if request.name not in _MODELS:
            context.abort(grpc.StatusCode.NOT_FOUND, f'unknown model {request.name}')
        _batch, inputs, outputs = _MODELS[request.name]
        response = service_pb2.ModelMetadataResponse(name=request.name, versions=['1'])
        for spec in inputs:
            response.inputs.add(name=spec.name, datatype=spec.datatype, shape=spec.shape)
        for spec in outputs:
            response.outputs.add(name=spec.name, datatype=spec.datatype, shape=spec.shape)
        return response

    def ModelConfig(
        self, request: service_pb2.ModelConfigRequest, context: grpc.ServicerContext
    ) -> service_pb2.ModelConfigResponse:
        if request.name not in _MODELS:
            context.abort(grpc.StatusCode.NOT_FOUND, f'unknown model {request.name}')
        max_batch, inputs, outputs = _MODELS[request.name]
        config = model_config_pb2.ModelConfig(name=request.name, max_batch_size=max_batch)
        for specs, target in ((inputs, config.input), (outputs, config.output)):
            for spec in specs:
                target.add(
                    name=spec.name,
                    data_type=_CONFIG_TYPES[spec.datatype],
                    dims=spec.shape[1:] if max_batch else spec.shape,
                )
        return service_pb2.ModelConfigResponse(config=config)

    def RepositoryModelLoad(
        self, request: service_pb2.RepositoryModelLoadRequest, context: grpc.ServicerContext
    ) -> service_pb2.RepositoryModelLoadResponse:
        if request.model_name not in _MODELS:
            context.abort(grpc.StatusCode.NOT_FOUND, f'unknown model {request.model_name}')
        return service_pb2.RepositoryModelLoadResponse()

    def ModelInfer(
        self, request: service_pb2.ModelInferRequest, context: grpc.ServicerContext
    ) -> service_pb2.ModelInferResponse:
        name = request.model_name
        if name == 'yolov11_small_trt_end2end':
            return _as_response(request, {'output0': _fixed(_detect(_array(request, 0)))})
        if name == 'pe_image_encoder':
            return _as_response(
                request, {'image_embeddings': _fixed(_image_embeddings(_array(request, 0)))}
            )
        if name == 'pe_text_encoder':
            return _as_response(
                request, {'text_embeddings': _fixed(_text_embeddings(_array(request, 0)))}
            )
        if name == 'ocr_pipeline':
            return _as_response(request, _ocr())
        context.abort(grpc.StatusCode.UNIMPLEMENTED, f'fake-triton serves no model {name}')
        raise AssertionError('unreachable')


class _Http(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == '/v2/health/ready':
            self._send(200, b'{}')
        elif (name := _model_of(self.path, '/v2/models/', '/config')) in _MODELS:
            self._send(200, json.dumps(_config_json(name)).encode())
        else:
            self._send(404, b'{}')

    def do_POST(self) -> None:
        if self.path == '/v2/repository/index':
            self._send(200, json.dumps(_repository_index()).encode())
        elif _model_of(self.path, '/v2/repository/models/', '/load') in _MODELS:
            self._send(200, b'{}')
        else:
            self._send(404, b'{}')

    def log_message(self, *_args: object) -> None:
        return None


def main() -> None:
    http = ThreadingHTTPServer(('0.0.0.0', 8000), _Http)  # nosec B104
    threading.Thread(target=http.serve_forever, daemon=True).start()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(_Models(), server)
    server.add_insecure_port('0.0.0.0:8001')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
