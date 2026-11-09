"""The PE text encoder is Triton only.

:class:`src.clients.pe_encoder.PEEncoder` binds Triton's shared ``pe_text_encoder`` at warm-up,
and when that model is not ready it refuses (``TextModelNotReady``) instead of loading an
in-process backend: no ONNX Runtime session, no PyTorch checkpoint. The tokenizer and the Triton
client are fakes, so everything here runs without weights, a GPU or ``perception_models``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.clients.pe_encoder import PE_EMBEDDING_DIM, PEEncoder, TextModelNotReady
from src.clients.pe_text_backends import (
    PE_TEXT_INPUT,
    PE_TEXT_OUTPUT,
    PE_TEXT_TRITON_MODEL,
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

    def infer(
        self,
        model_name: str,
        inputs: list[Any],
        outputs: list[Any],
        client_timeout: float | None = None,
    ) -> Any:
        assert client_timeout, 'every sync Triton call carries a deadline'
        (inp,) = inputs
        self.infer_calls.append((model_name, inp.shape(), inp.datatype()))
        if self.fail_infer:
            raise ConnectionError('triton went away')
        assert inp.name() == PE_TEXT_INPUT
        assert outputs[0].name() == PE_TEXT_OUTPUT
        batch = inp.shape()[0]
        return _FakeInferResult(np.ones((batch, PE_EMBEDDING_DIM), dtype=np.float32))


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


def _encoder(
    triton: _FakeTritonClient | Exception | None, monkeypatch: pytest.MonkeyPatch
) -> PEEncoder:
    """A PEEncoder wired to a fake Triton client; the tokenizer never imports ``core``."""

    def factory() -> Any:
        if isinstance(triton, Exception):
            raise triton
        if triton is None:
            raise ConnectionError('no triton in this test')
        return triton

    enc = PEEncoder(triton_client_factory=factory)
    monkeypatch.setattr(enc, '_load_tokenizer', lambda: _fake_tokenizer)
    return enc


# =============================================================================
# Warm-up
# =============================================================================


class TestWarmUp:
    def test_a_ready_text_model_binds_the_triton_backend(self, monkeypatch) -> None:
        triton = _FakeTritonClient(ready=True)
        enc = _encoder(triton, monkeypatch)
        enc.warm_text_encoder()
        assert enc.text_status()['ready']
        assert triton.ready_checks == [PE_TEXT_TRITON_MODEL]
        out = enc.encode_text(['a white truck'])
        assert out.shape == (1, PE_EMBEDDING_DIM)
        model, shape, dtype = triton.infer_calls[0]
        assert (model, dtype) == (PE_TEXT_TRITON_MODEL, 'INT64')
        assert shape == [1, 15]  # SOT, the 13 characters, EOT: trimmed to the longest EOT

    def test_a_text_model_that_is_not_ready_refuses_and_names_it(self, monkeypatch) -> None:
        enc = _encoder(_FakeTritonClient(ready=False), monkeypatch)
        with pytest.raises(TextModelNotReady, match=PE_TEXT_TRITON_MODEL):
            enc.warm_text_encoder()
        assert enc.text_status()['ready'] is False

    def test_an_unreachable_triton_refuses_the_same_way(self, monkeypatch) -> None:
        enc = _encoder(ConnectionError('refused'), monkeypatch)
        with pytest.raises(TextModelNotReady, match=PE_TEXT_TRITON_MODEL):
            enc.warm_text_encoder()

    def test_the_tokenizer_is_not_loaded_while_the_model_is_not_ready(self, monkeypatch) -> None:
        enc = _encoder(_FakeTritonClient(ready=False), monkeypatch)
        loaded: list[bool] = []
        monkeypatch.setattr(enc, '_load_tokenizer', lambda: loaded.append(True))
        with pytest.raises(TextModelNotReady):
            enc.warm_text_encoder()
        assert loaded == []

    def test_ensure_text_ready_retries_once_triton_is_up(self, monkeypatch) -> None:
        triton = _FakeTritonClient(ready=False)
        enc = _encoder(triton, monkeypatch)
        assert enc.ensure_text_ready() is False
        triton.ready = True
        assert enc.ensure_text_ready() is True
        assert enc.text_status()['ready']

    def test_the_model_name_comes_from_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv('OP_PE_TEXT_TRITON_MODEL', 'pe_text_encoder_v2')
        assert PEEncoder().text_model == 'pe_text_encoder_v2'


class TestTritonRuntime:
    def test_a_failed_triton_call_is_not_swallowed(self, monkeypatch) -> None:
        enc = _encoder(_FakeTritonClient(ready=True, fail_infer=True), monkeypatch)
        enc.warm_text_encoder()
        with pytest.raises(ConnectionError):
            enc.encode_text(['a white truck'])
        assert enc.text_status()['ready'], 'a transient failure does not unbind the backend'


# =============================================================================
# Health endpoint
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

    def test_reports_the_text_model_once_bound(self, monkeypatch) -> None:
        enc = _encoder(_FakeTritonClient(ready=True), monkeypatch)
        enc.warm_text_encoder()
        resp = self._client(enc).get('/health/pe_text')
        assert resp.status_code == 200
        body = resp.json()
        assert body['ready'] is True
        assert body['triton_model'] == PE_TEXT_TRITON_MODEL
        assert 'onnx_path' not in body

    def test_cold_encoder_is_503(self) -> None:
        resp = self._client(PEEncoder()).get('/health/pe_text')
        assert resp.status_code == 503
        assert resp.json()['ready'] is False

    def test_missing_encoder_is_503(self) -> None:
        resp = self._client(None).get('/health/pe_text')
        assert resp.status_code == 503
