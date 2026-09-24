"""Backend selection + fallback for the PE text encoder.

:class:`src.clients.pe_encoder.PEEncoder` picks one of three text backends
at warm-up — in-process ONNX Runtime when ``OP_PE_TEXT_ONNX_PATH`` exists,
Triton's ``pe_text_encoder`` when it reports ready, PyTorch eager otherwise
— and drops to in-process encoding for good if Triton fails mid-flight.

Everything here runs on CI without weights, a GPU or ``perception_models``:
the ONNX backend is exercised against a genuine, tiny ONNX graph with the
real tensor contract (``text_tokens`` INT64 ``[B, T]`` ->
``text_embeddings`` FP32 ``[B, 1024]``) on the ORT CPU provider; the
tokenizer, the PyTorch loader and the Triton client are fakes. The last
test runs the real exported graph against the real PyTorch model and skips
cleanly where those are unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.clients.pe_encoder import (
    PE_EMBEDDING_DIM,
    PE_TEXT_INPUT,
    PE_TEXT_OUTPUT,
    PE_TEXT_TRITON_MODEL,
    PEEncoder,
    trim_text_tokens,
)


SOT, EOT = 49406, 49407


# =============================================================================
# Fakes
# =============================================================================


def _fake_tokenizer(queries: list[str]) -> np.ndarray:
    out = np.zeros((len(queries), 32), dtype=np.int64)
    for row, query in enumerate(queries):
        ids = [SOT, *(100 + (ord(c) % 1000) for c in query[:30]), EOT]
        out[row, : len(ids)] = ids
    return out


def _expected_row(tokens_row: np.ndarray) -> np.ndarray:
    """What the stub graph computes for one token row (pre-normalization)."""
    return np.full(PE_EMBEDDING_DIM, float(tokens_row.sum()), dtype=np.float32) * _PROJ


# Non-uniform projection so rows with different token sums differ in
# direction only through the sign — enough to tell rows apart after L2.
_PROJ = np.linspace(-1.0, 1.0, PE_EMBEDDING_DIM, dtype=np.float32)


def _write_stub_graph(
    path: Path,
    *,
    input_name: str = PE_TEXT_INPUT,
    output_name: str = PE_TEXT_OUTPUT,
) -> Path:
    """``text_tokens[B, T] int64 -> text_embeddings[B, 1024]`` via sum x proj."""
    pytest.importorskip('onnxruntime')
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    tokens = helper.make_tensor_value_info(input_name, TensorProto.INT64, ['batch', 'tokens'])
    emb = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, ['batch', PE_EMBEDDING_DIM])
    proj = numpy_helper.from_array(_PROJ.reshape(1, PE_EMBEDDING_DIM), name='proj')
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name='axes')
    graph = helper.make_graph(
        [
            helper.make_node('Cast', [input_name], ['as_float'], to=TensorProto.FLOAT),
            helper.make_node('ReduceSum', ['as_float', 'axes'], ['summed'], keepdims=1),
            helper.make_node('MatMul', ['summed', 'proj'], [output_name]),
        ],
        'pe_text_stub',
        [tokens],
        [emb],
        initializer=[proj, axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


class _FakeTorchBackend:
    name = 'torch'

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, tokens: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.stack([_expected_row(r) for r in tokens])


class _FakeInferResult:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def as_numpy(self, name: str) -> np.ndarray:
        assert name == PE_TEXT_OUTPUT
        return self._array


class _FakeTritonClient:
    def __init__(self, *, ready: bool = True, fail_infer: bool = False) -> None:
        self.ready = ready
        self.fail_infer = fail_infer
        self.ready_checks: list[str] = []
        self.infer_calls: list[tuple[str, list[int], str]] = []

    def is_model_ready(self, model_name: str) -> bool:
        self.ready_checks.append(model_name)
        return self.ready

    def infer(self, model_name: str, inputs: list[Any], outputs: list[Any]) -> Any:
        (inp,) = inputs
        self.infer_calls.append((model_name, inp.shape(), inp.datatype()))
        if self.fail_infer:
            raise ConnectionError('triton went away')
        assert inp.name() == PE_TEXT_INPUT
        assert outputs[0].name() == PE_TEXT_OUTPUT
        batch = inp.shape()[0]
        return _FakeInferResult(np.ones((batch, PE_EMBEDDING_DIM), dtype=np.float32))


def _encoder(
    tmp_path: Path,
    *,
    backend: str = 'auto',
    onnx_file: Path | None = None,
    triton: _FakeTritonClient | Exception | None = None,
    torch_backend: _FakeTorchBackend | None = None,
    monkeypatch: pytest.MonkeyPatch,
) -> PEEncoder:
    """A PEEncoder wired to fakes; the tokenizer + torch loader never import core."""

    def factory() -> Any:
        if isinstance(triton, Exception):
            raise triton
        if triton is None:
            raise ConnectionError('no triton in this test')
        return triton

    enc = PEEncoder(
        text_backend=backend,
        text_onnx_path=onnx_file or tmp_path / 'missing.onnx',
        triton_client_factory=factory,
    )
    monkeypatch.setattr(enc, '_load_tokenizer', lambda: _fake_tokenizer)

    def load_torch() -> Any:
        if torch_backend is None:
            raise ModuleNotFoundError("No module named 'core'")
        return torch_backend

    monkeypatch.setattr(enc, '_load_torch_backend', load_torch)
    return enc


# =============================================================================
# trim_text_tokens
# =============================================================================


class TestTrimTextTokens:
    def test_trims_to_the_batch_longest_eot(self) -> None:
        tokens = _fake_tokenizer(['ab', 'abcd'])
        out = trim_text_tokens(tokens)
        assert out.shape == (2, 6)  # SOT + 4 + EOT
        assert out[0, 3] == EOT
        assert out[1, 5] == EOT
        assert out.flags['C_CONTIGUOUS']
        np.testing.assert_array_equal(out, tokens[:, :6])

    def test_a_truncated_query_keeps_the_full_context(self) -> None:
        tokens = np.zeros((1, 32), dtype=np.int64)
        tokens[0, :31] = 500
        tokens[0, 31] = EOT
        assert trim_text_tokens(tokens).shape == (1, 32)

    def test_degenerate_inputs_pass_through(self) -> None:
        assert trim_text_tokens(np.zeros((0, 32), dtype=np.int64)).shape == (0, 32)
        assert trim_text_tokens(np.zeros((2, 32), dtype=np.int64)).shape == (2, 1)


# =============================================================================
# Selection
# =============================================================================


class TestBackendSelection:
    def test_auto_prefers_onnx_when_the_file_exists(self, tmp_path, monkeypatch) -> None:
        onnx_file = _write_stub_graph(tmp_path / 'pe_text.onnx')
        triton = _FakeTritonClient()
        torch_backend = _FakeTorchBackend()
        enc = _encoder(
            tmp_path,
            onnx_file=onnx_file,
            triton=triton,
            torch_backend=torch_backend,
            monkeypatch=monkeypatch,
        )

        enc.warm_text_encoder()

        assert enc.text_ready
        assert enc.text_backend == 'onnx'
        assert triton.ready_checks == []
        assert torch_backend.calls == 0

    def test_onnx_backend_runs_the_graph_on_trimmed_tokens(self, tmp_path, monkeypatch) -> None:
        onnx_file = _write_stub_graph(tmp_path / 'pe_text.onnx')
        enc = _encoder(tmp_path, onnx_file=onnx_file, monkeypatch=monkeypatch)
        enc.warm_text_encoder()

        out = enc.encode_text(['red sedan', 'x'])

        tokens = trim_text_tokens(_fake_tokenizer(['red sedan', 'x']))
        expected = np.stack([_expected_row(r) for r in tokens])
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        assert out.shape == (2, PE_EMBEDDING_DIM)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

    def test_auto_uses_triton_when_no_onnx_and_model_ready(self, tmp_path, monkeypatch) -> None:
        triton = _FakeTritonClient(ready=True)
        torch_backend = _FakeTorchBackend()
        enc = _encoder(
            tmp_path, triton=triton, torch_backend=torch_backend, monkeypatch=monkeypatch
        )

        enc.warm_text_encoder()
        out = enc.encode_text(['red sedan'])

        assert enc.text_backend == 'triton'
        assert triton.ready_checks == [PE_TEXT_TRITON_MODEL]
        # SOT + 9 chars + EOT, trimmed, int64.
        assert triton.infer_calls == [(PE_TEXT_TRITON_MODEL, [1, 11], 'INT64')]
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0], atol=1e-6)
        # Triton path never loads the in-process model up front.
        assert torch_backend.calls == 0

    def test_auto_falls_back_to_torch_when_nothing_else(self, tmp_path, monkeypatch) -> None:
        torch_backend = _FakeTorchBackend()
        enc = _encoder(
            tmp_path,
            triton=_FakeTritonClient(ready=False),
            torch_backend=torch_backend,
            monkeypatch=monkeypatch,
        )
        enc.warm_text_encoder()
        enc.encode_text(['red sedan'])
        assert enc.text_backend == 'torch'
        assert torch_backend.calls == 1

    def test_unreachable_triton_is_not_an_error(self, tmp_path, monkeypatch) -> None:
        enc = _encoder(
            tmp_path,
            triton=ConnectionError('refused'),
            torch_backend=_FakeTorchBackend(),
            monkeypatch=monkeypatch,
        )
        enc.warm_text_encoder()
        assert enc.text_backend == 'torch'

    def test_missing_onnx_file_falls_back_under_auto(self, tmp_path, monkeypatch) -> None:
        enc = _encoder(
            tmp_path,
            onnx_file=tmp_path / 'not_exported_yet.onnx',
            torch_backend=_FakeTorchBackend(),
            monkeypatch=monkeypatch,
        )
        enc.warm_text_encoder()
        assert enc.text_backend == 'torch'
        assert enc.text_status()['onnx_path_exists'] is False

    def test_unloadable_onnx_file_falls_back_under_auto(self, tmp_path, monkeypatch) -> None:
        bad = tmp_path / 'corrupt.onnx'
        bad.write_bytes(b'not an onnx graph')
        enc = _encoder(
            tmp_path, onnx_file=bad, torch_backend=_FakeTorchBackend(), monkeypatch=monkeypatch
        )
        enc.warm_text_encoder()
        assert enc.text_backend == 'torch'

    def test_pinned_onnx_with_missing_file_raises(self, tmp_path, monkeypatch) -> None:
        enc = _encoder(
            tmp_path, backend='onnx', torch_backend=_FakeTorchBackend(), monkeypatch=monkeypatch
        )
        with pytest.raises(FileNotFoundError, match='export_pe_text_encoder'):
            enc.warm_text_encoder()
        assert enc.text_ready is False

    def test_pinned_onnx_with_wrong_tensor_names_raises(self, tmp_path, monkeypatch) -> None:
        onnx_file = _write_stub_graph(tmp_path / 'wrong.onnx', input_name='input_ids')
        enc = _encoder(tmp_path, backend='onnx', onnx_file=onnx_file, monkeypatch=monkeypatch)
        with pytest.raises(ValueError, match='text_tokens'):
            enc.warm_text_encoder()

    def test_pinned_torch_ignores_an_existing_onnx_file(self, tmp_path, monkeypatch) -> None:
        onnx_file = _write_stub_graph(tmp_path / 'pe_text.onnx')
        enc = _encoder(
            tmp_path,
            backend='torch',
            onnx_file=onnx_file,
            triton=_FakeTritonClient(),
            torch_backend=_FakeTorchBackend(),
            monkeypatch=monkeypatch,
        )
        enc.warm_text_encoder()
        assert enc.text_backend == 'torch'

    def test_pinned_torch_without_perception_models_raises(self, tmp_path, monkeypatch) -> None:
        enc = _encoder(tmp_path, backend='torch', monkeypatch=monkeypatch)
        with pytest.raises(ModuleNotFoundError):
            enc.warm_text_encoder()
        assert enc.text_ready is False

    def test_pinned_triton_not_ready_uses_local(self, tmp_path, monkeypatch) -> None:
        onnx_file = _write_stub_graph(tmp_path / 'pe_text.onnx')
        enc = _encoder(
            tmp_path,
            backend='triton',
            onnx_file=onnx_file,
            triton=_FakeTritonClient(ready=False),
            monkeypatch=monkeypatch,
        )
        enc.warm_text_encoder()
        assert enc.text_backend == 'onnx'

    def test_pinned_triton_prefers_triton_over_a_local_onnx(self, tmp_path, monkeypatch) -> None:
        onnx_file = _write_stub_graph(tmp_path / 'pe_text.onnx')
        enc = _encoder(
            tmp_path,
            backend='triton',
            onnx_file=onnx_file,
            triton=_FakeTritonClient(ready=True),
            monkeypatch=monkeypatch,
        )
        enc.warm_text_encoder()
        assert enc.text_backend == 'triton'

    def test_unknown_backend_name_is_rejected(self, tmp_path, monkeypatch) -> None:
        enc = _encoder(tmp_path, backend='tensorrt', monkeypatch=monkeypatch)
        with pytest.raises(ValueError, match='OP_PE_TEXT_BACKEND'):
            enc.warm_text_encoder()

    def test_environment_configures_the_defaults(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv('OP_PE_TEXT_BACKEND', ' ONNX ')
        monkeypatch.setenv('OP_PE_TEXT_ONNX_PATH', str(tmp_path / 'x.onnx'))
        monkeypatch.setenv('OP_PE_TEXT_TRITON_MODEL', 'pe_text_encoder_v2')
        monkeypatch.setenv('OP_PE_TEXT_ORT_THREADS', '4')
        status = PEEncoder().text_status()
        assert status['requested_backend'] == 'onnx'
        assert status['onnx_path'] == str(tmp_path / 'x.onnx')
        assert status['triton_model'] == 'pe_text_encoder_v2'
        assert PEEncoder()._ort_threads == 4

    def test_default_onnx_path_is_the_api_container_mount(self, monkeypatch) -> None:
        monkeypatch.delenv('OP_PE_TEXT_ONNX_PATH', raising=False)
        assert PEEncoder().text_status()['onnx_path'] == '/app/pytorch_models/pe_text_encoder.onnx'


# =============================================================================
# Runtime fallback
# =============================================================================


class TestTritonRuntimeFallback:
    def test_failed_triton_call_switches_to_local_for_good(self, tmp_path, monkeypatch) -> None:
        onnx_file = _write_stub_graph(tmp_path / 'pe_text.onnx')
        triton = _FakeTritonClient(ready=True, fail_infer=True)
        enc = _encoder(
            tmp_path, backend='triton', onnx_file=onnx_file, triton=triton, monkeypatch=monkeypatch
        )
        enc.warm_text_encoder()
        assert enc.text_backend == 'triton'

        out = enc.encode_text(['red sedan'])
        enc.encode_text(['blue truck'])

        assert out.shape == (1, PE_EMBEDDING_DIM)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0], atol=1e-6)
        assert enc.text_backend == 'onnx'
        assert enc.text_status()['triton_fallbacks'] == 1
        # Only the first query ever reached Triton — no per-request flapping.
        assert len(triton.infer_calls) == 1

    def test_fallback_loads_torch_lazily_when_no_onnx(self, tmp_path, monkeypatch) -> None:
        torch_backend = _FakeTorchBackend()
        enc = _encoder(
            tmp_path,
            triton=_FakeTritonClient(ready=True, fail_infer=True),
            torch_backend=torch_backend,
            monkeypatch=monkeypatch,
        )
        enc.warm_text_encoder()
        assert torch_backend.calls == 0

        enc.encode_text(['red sedan'])

        assert enc.text_backend == 'torch'
        assert torch_backend.calls == 1

    def test_a_failing_local_backend_is_not_swallowed(self) -> None:
        class Broken:
            name = 'onnx'

            def encode(self, tokens: np.ndarray) -> np.ndarray:
                raise RuntimeError(f'ORT kernel failure on {tokens.shape}')

        enc = PEEncoder()
        enc._text_tokenizer = _fake_tokenizer
        enc._text_backend = Broken()
        enc._text_ready = True
        with pytest.raises(RuntimeError, match='ORT kernel failure'):
            enc.encode_text(['red sedan'])
        # Nothing was cached for the failed query.
        assert enc.text_cache_info().currsize == 0


# =============================================================================
# GET /health/pe_text
# =============================================================================


class TestHealthEndpoint:
    @staticmethod
    def _client(encoder: Any) -> Any:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.routers.health import router

        app = FastAPI()
        app.include_router(router)
        if encoder is not None:
            app.state.pe_encoder = encoder
        return TestClient(app)

    def test_reports_the_active_backend(self, tmp_path, monkeypatch) -> None:
        onnx_file = _write_stub_graph(tmp_path / 'pe_text.onnx')
        enc = _encoder(tmp_path, onnx_file=onnx_file, monkeypatch=monkeypatch)
        enc.warm_text_encoder()
        resp = self._client(enc).get('/health/pe_text')
        assert resp.status_code == 200
        body = resp.json()
        assert body['ready'] is True
        assert body['backend'] == 'onnx'
        assert body['onnx_path'] == str(onnx_file)

    def test_cold_encoder_is_503(self, tmp_path) -> None:
        resp = self._client(PEEncoder(text_onnx_path=tmp_path / 'x.onnx')).get('/health/pe_text')
        assert resp.status_code == 503
        assert resp.json()['backend'] is None

    def test_missing_encoder_is_503(self) -> None:
        resp = self._client(None).get('/health/pe_text')
        assert resp.status_code == 503


# =============================================================================
# Real weights (skipped unless the checkpoint, perception_models and an
# exported graph are all present — e.g. inside the API image)
# =============================================================================


def test_real_onnx_matches_pytorch_eager() -> None:
    """ORT on the exported graph vs PyTorch eager, through PEEncoder itself.

    Needs ``perception_models`` + the PE checkpoint (HF cache) and the graph
    at ``OP_PE_TEXT_ONNX_PATH``. The exporter's own parity gate runs the
    same comparison at export time; this re-checks the *client* wiring
    (tokenizer, EOT trimming, normalization) end to end.
    """
    pytest.importorskip('onnxruntime')
    try:
        import core.vision_encoder.tokenizer  # noqa: F401
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        pytest.skip(f'perception_models/torch not installed ({exc})')
    onnx_path = os.environ.get('OP_PE_TEXT_ONNX_PATH', '/app/pytorch_models/pe_text_encoder.onnx')
    if not Path(onnx_path).is_file():
        pytest.skip(f'no exported PE text graph at {onnx_path}')

    prompts = [
        'a photo of a white pickup truck',
        'red sedan',
        'close-up of a license plate',
        'café storefront with a neon sign',
        'two dogs playing in the snow on a sunny winter afternoon near a frozen lake '
        'with pine trees and mountains in the background under a clear blue sky',
    ]
    onnx_enc = PEEncoder(text_backend='onnx', text_onnx_path=onnx_path)
    onnx_enc.warm_text_encoder()
    try:
        torch_enc = PEEncoder(text_backend='torch')
        torch_enc.warm_text_encoder()
    except Exception as exc:  # checkpoint not in the HF cache / offline
        pytest.skip(f'PE checkpoint unavailable ({exc})')

    for batch in ([prompts[0]], prompts):
        a = onnx_enc._encode_uncached(batch)
        b = torch_enc._encode_uncached(batch)
        cos = (a * b).sum(axis=1)
        assert cos.min() >= 0.9999, cos
