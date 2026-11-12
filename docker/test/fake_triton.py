"""A Triton stand-in for the live harness: liveness, readiness, a model repository index and
deterministic models.

The API refuses to start without a live Triton (src/app/startup_gate.py, owner decision D-69:
no degraded mode). The harness verifies the write paths without a GPU, so this answers
``ServerLive`` and ``ServerReady`` over Triton's gRPC protocol, and over HTTP the readiness probe
and the repository index (the config vocabulary lists the served detectors from it).

These models answer over gRPC so the ingest, search, face, region-text and OCR routes run end to
end (the fixed outputs are in ``_MODELS`` and the helpers below it): the fused detector (one box
per picture), the PE image encoder (a vector from the picture's coarse colour layout, so a
picture is its own nearest neighbour), the PE text encoder (a vector from the token ids, so it
cannot read a query), the OCR pipeline (one fixed line), and the core face and CLIP models: SCRFD
(one face with five landmarks per picture, in the raw FPN tensors the API decodes), ArcFace (a
512-d vector from the aligned crop's bytes) and MobileCLIP (image: coarse colour layout; text: token
hash). Tensor names, dtypes and shapes follow ``models/<name>/config.pbtxt``, and the wire is the
KServe v2 inference protocol that Triton implements
(https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md; Triton's
gRPC service is ``inference.GRPCInferenceService``).

A model a training promote wrote into the repository (``FAKE_TRITON_MODEL_REPO``, a directory
holding ``<name>/<version>/model.onnx``) is listed once loaded and answers like the detector, over
gRPC and over the HTTP ``/v2/models/<name>/infer`` call the promote's warm-up makes. Every other
model and call stays unimplemented: a code path that needs other inference still fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
from concurrent import futures
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple

import grpc
import numpy as np
from tritonclient.grpc import model_config_pb2, service_pb2, service_pb2_grpc
from tritonclient.utils import np_to_triton_dtype, triton_to_np_dtype


#: The text the OCR stand-in reads on every image, and its score.
FAKE_OCR_TEXT = 'FAKE-PLATE-0042'
EMBED_DIM = 1024
CLIP_DIM = 512
FACE_DIM = 512
MAX_DET = 300
#: Where a training promote writes models; the stand-in lists a loaded one from here.
MODEL_REPO = Path(os.environ.get('FAKE_TRITON_MODEL_REPO', '/triton-models'))
_loaded_from_repo: set[str] = set()
#: Triton accepts messages up to INT32_MAX; grpc's default 4 MiB refuses one SCRFD frame (4.9 MB).
GRPC_OPTIONS = [
    ('grpc.max_receive_message_length', 2**31 - 1),
    ('grpc.max_send_message_length', 2**31 - 1),
]


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


#: SCRFD's feature-pyramid strides (src/utils/scrfd_decode.py: two anchors per position).
_STRIDES = (8, 16, 32)

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
    'scrfd_10g_bnkps': (
        64,
        [_tensor('input.1', 'FP32', [-1, 3, 640, 640])],
        [
            *(_tensor(f'score_{s}', 'FP32', [-1, (640 // s) ** 2 * 2, 1]) for s in _STRIDES),
            *(_tensor(f'bbox_{s}', 'FP32', [-1, (640 // s) ** 2 * 2, 4]) for s in _STRIDES),
            *(_tensor(f'kps_{s}', 'FP32', [-1, (640 // s) ** 2 * 2, 10]) for s in _STRIDES),
        ],
    ),
    'arcface_w600k_r50': (
        128,
        [_tensor('input', 'FP32', [-1, 3, 112, 112])],
        [_tensor('output', 'FP32', [-1, FACE_DIM])],
    ),
    'mobileclip2_s2_image_encoder': (
        64,
        [_tensor('images', 'FP32', [-1, 3, 256, 256])],
        [_tensor('image_embeddings', 'FP32', [-1, CLIP_DIM])],
    ),
    'mobileclip2_s2_text_encoder': (
        64,
        [_tensor('text_tokens', 'INT64', [-1, 77])],
        [_tensor('text_embeddings', 'FP32', [-1, CLIP_DIM])],
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


#: What a promoted detector serves (models/yolov11_small_trt_end2end shape, uint8 in).
_PROMOTED_SPEC = _MODELS['yolov11_small_trt_end2end']


def _on_disk(name: str) -> bool:
    versions = MODEL_REPO / name
    return versions.is_dir() and any(versions.glob('[0-9]*/model.onnx'))


def _promoted() -> list[str]:
    """Models a promote wrote that Triton has been asked to load and has not unloaded."""
    return sorted(n for n in _loaded_from_repo if n not in _MODELS and _on_disk(n))


def _spec(name: str) -> tuple[int, list[_Tensor], list[_Tensor]] | None:
    if name in _MODELS:
        return _MODELS[name]
    return _PROMOTED_SPEC if name in _promoted() else None


def _repository_index() -> list[dict[str, str]]:
    return [
        {'name': name, 'version': '1', 'state': 'READY', 'reason': ''}
        for name in (*_MODELS, *_LISTED_ONLY, *_promoted())
    ]


def _load(name: str) -> bool:
    """Triton's explicit load: a built-in is always loaded; a promoted model needs its ONNX."""
    if name in _MODELS:
        return True
    if not _on_disk(name):
        return False
    _loaded_from_repo.add(name)
    return True


def _model_of(path: str, prefix: str, suffix: str) -> str:
    if path.startswith(prefix) and path.endswith(suffix):
        return path.removeprefix(prefix).removesuffix(suffix)
    return ''


def _config_json(name: str) -> dict[str, object]:
    """Triton's HTTP model-configuration document (the fields the API reads)."""
    spec = _spec(name)
    assert spec is not None
    max_batch, inputs, outputs = spec

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


def _image_embeddings(images: np.ndarray, dim: int) -> np.ndarray:
    """An 8x8 grid of mean colours (uint8 centred on mid-grey), tiled to ``dim`` dimensions."""
    n, c, h, w = images.shape
    grid = images.astype(np.float32).reshape(n, c, 8, h // 8, 8, w // 8).mean(axis=(3, 5))
    flat = (grid - (128.0 if images.dtype == np.uint8 else 0.0)).reshape(n, -1)
    return _unit(np.tile(flat, (1, dim // flat.shape[1] + 1))[:, :dim])


def _hashed_embeddings(rows: np.ndarray, dim: int) -> np.ndarray:
    """A deterministic unit vector per row, seeded from the row's bytes."""
    out = []
    for row in rows:
        seed = int.from_bytes(
            hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest()[:8], 'little'
        )
        out.append(np.random.default_rng(seed).standard_normal(dim))
    return _unit(np.asarray(out, dtype=np.float32))


def _scrfd(batch: int) -> dict[str, np.ndarray]:
    """One face per frame, centred: raw FPN tensors as ``decode_scrfd_outputs`` reads them.

    The anchor at stride 8, grid cell (40, 40), anchor 0 sits at pixel (320, 320). Distances are in
    stride units (the decoder multiplies by the stride): 5 -> 40 px each way, an 80 px box, and the
    five landmarks (eyes, nose, mouth corners) at offsets a real face has.
    """
    out: dict[str, np.ndarray] = {}
    for s in _STRIDES:
        n = (640 // s) ** 2 * 2
        out[f'score_{s}'] = np.zeros((batch, n, 1), dtype=np.float32)
        out[f'bbox_{s}'] = np.zeros((batch, n, 4), dtype=np.float32)
        out[f'kps_{s}'] = np.zeros((batch, n, 10), dtype=np.float32)
    anchor = (40 * 80 + 40) * 2
    out['score_8'][:, anchor, 0] = 0.95
    out['bbox_8'][:, anchor] = [5.0, 5.0, 5.0, 5.0]
    out['kps_8'][:, anchor] = [-2.5, -1.5, 2.5, -1.5, 0.0, 0.5, -2.0, 3.0, 2.0, 3.0]
    return out


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
        return service_pb2.ModelReadyResponse(ready=_spec(request.name) is not None)

    def ModelMetadata(
        self, request: service_pb2.ModelMetadataRequest, context: grpc.ServicerContext
    ) -> service_pb2.ModelMetadataResponse:
        served = _spec(request.name)
        if served is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f'unknown model {request.name}')
            raise AssertionError('unreachable')
        _batch, inputs, outputs = served
        response = service_pb2.ModelMetadataResponse(name=request.name, versions=['1'])
        for spec in inputs:
            response.inputs.add(name=spec.name, datatype=spec.datatype, shape=spec.shape)
        for spec in outputs:
            response.outputs.add(name=spec.name, datatype=spec.datatype, shape=spec.shape)
        return response

    def ModelConfig(
        self, request: service_pb2.ModelConfigRequest, context: grpc.ServicerContext
    ) -> service_pb2.ModelConfigResponse:
        served = _spec(request.name)
        if served is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f'unknown model {request.name}')
            raise AssertionError('unreachable')
        max_batch, inputs, outputs = served
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
        if not _load(request.model_name):
            context.abort(grpc.StatusCode.NOT_FOUND, f'unknown model {request.model_name}')
        return service_pb2.RepositoryModelLoadResponse()

    def ModelInfer(
        self, request: service_pb2.ModelInferRequest, context: grpc.ServicerContext
    ) -> service_pb2.ModelInferResponse:
        name = request.model_name
        if name == 'yolov11_small_trt_end2end' or name in _promoted():
            return _as_response(request, {'output0': _fixed(_detect(_array(request, 0)))})
        if name == 'pe_image_encoder':
            return _as_response(
                request,
                {'image_embeddings': _fixed(_image_embeddings(_array(request, 0), EMBED_DIM))},
            )
        if name == 'pe_text_encoder':
            return _as_response(
                request,
                {'text_embeddings': _fixed(_hashed_embeddings(_array(request, 0), EMBED_DIM))},
            )
        if name == 'mobileclip2_s2_image_encoder':
            return _as_response(
                request,
                {'image_embeddings': _fixed(_image_embeddings(_array(request, 0), CLIP_DIM))},
            )
        if name == 'mobileclip2_s2_text_encoder':
            return _as_response(
                request,
                {'text_embeddings': _fixed(_hashed_embeddings(_array(request, 0), CLIP_DIM))},
            )
        if name == 'arcface_w600k_r50':
            return _as_response(
                request, {'output': _fixed(_hashed_embeddings(_array(request, 0), FACE_DIM))}
            )
        if name == 'scrfd_10g_bnkps':
            raw = _scrfd(int(_array(request, 0).shape[0]))
            return _as_response(request, {k: _fixed(v) for k, v in raw.items()})
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
        elif _spec(name := _model_of(self.path, '/v2/models/', '/config')) is not None:
            self._send(200, json.dumps(_config_json(name)).encode())
        else:
            self._send(404, b'{}')

    def do_POST(self) -> None:
        if self.path == '/v2/repository/index':
            self._send(200, json.dumps(_repository_index()).encode())
        elif _model_of(self.path, '/v2/repository/models/', '/load') != '':
            ok = _load(_model_of(self.path, '/v2/repository/models/', '/load'))
            self._send(200 if ok else 400, b'{}')
        elif (name := _model_of(self.path, '/v2/repository/models/', '/unload')) != '':
            _loaded_from_repo.discard(name)
            self._send(200, b'{}')
        elif (name := _model_of(self.path, '/v2/models/', '/infer')) in _promoted():
            self._infer_json(name)
        else:
            self._send(404, b'{}')

    def _infer_json(self, name: str) -> None:
        """KServe v2 JSON infer: ``inputs[].{name, shape, datatype, data}`` in, ``outputs`` out."""
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
        shape = body['inputs'][0]['shape']
        out = _detect(np.zeros(shape, dtype=np.uint8))
        reply = {
            'model_name': name,
            'model_version': '1',
            'outputs': [
                {
                    'name': 'output0',
                    'datatype': 'FP32',
                    'shape': list(out.shape),
                    'data': out.flatten().tolist(),
                }
            ],
        }
        self._send(200, json.dumps(reply).encode())

    def log_message(self, *_args: object) -> None:
        return None


def main() -> None:
    http = ThreadingHTTPServer(('0.0.0.0', 8000), _Http)  # nosec B104
    threading.Thread(target=http.serve_forever, daemon=True).start()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=GRPC_OPTIONS)
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(_Models(), server)
    server.add_insecure_port('0.0.0.0:8001')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
