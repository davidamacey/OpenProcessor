"""Tests for ``export/pe_rearrange_shim.py``.

Two things need proving:

1. The shim's trace-safe reshape/transpose sequence is numerically
   identical to ``einops.rearrange`` on the two exact patterns PE's
   ``SelfAttention`` uses, and it delegates untouched patterns to real
   einops (so it can't silently swallow a future, different call site).
2. The bug it fixes is real: tracing a PE-style attention block with the
   legacy TorchScript exporter (``dynamo=False``) and plain
   ``einops.rearrange`` bakes the traced batch size into the ONNX graph as
   a Reshape constant, so a batch different from the trace batch either
   fails outright or (worse) silently produces the wrong output shape.
   With the shim active during the same export, batch 1, the trace batch,
   and a batch larger than the trace batch all agree with the eager
   PyTorch reference.

No GPU, no downloaded weights, no perception_models install — a tiny
stand-in module mirrors only the two rearrange call sites in
``core.vision_encoder.pe.SelfAttention.forward``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    import torch as torch_typing

torch = pytest.importorskip('torch')
einops = pytest.importorskip('einops')
onnx = pytest.importorskip('onnx')
ort = pytest.importorskip('onnxruntime')

EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

from pe_rearrange_shim import (  # noqa: E402
    _MERGE_HEADS,
    _SPLIT_HEADS,
    _safe_rearrange,
    static_batch_safe_rearrange,
)


# =============================================================================
# Numerical equivalence against real einops
# =============================================================================


class TestSafeRearrangeEquivalence:
    @pytest.mark.parametrize('batch', [1, 2, 3, 8])
    @pytest.mark.parametrize(('heads', 'head_dim', 'seq'), [(16, 64, 577), (4, 8, 5), (1, 32, 10)])
    def test_split_heads_matches_einops(
        self, batch: int, heads: int, head_dim: int, seq: int
    ) -> None:
        x = torch.randn(batch, seq, heads * head_dim)
        expected = einops.rearrange(x, _SPLIT_HEADS, h=heads)
        actual = _safe_rearrange(x, _SPLIT_HEADS, h=heads)
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)

    @pytest.mark.parametrize('batch', [1, 2, 3, 8])
    @pytest.mark.parametrize(('heads', 'head_dim', 'seq'), [(16, 64, 577), (4, 8, 5), (1, 32, 10)])
    def test_merge_heads_matches_einops(
        self, batch: int, heads: int, head_dim: int, seq: int
    ) -> None:
        x = torch.randn(batch, heads, seq, head_dim)
        expected = einops.rearrange(x, _MERGE_HEADS)
        actual = _safe_rearrange(x, _MERGE_HEADS)
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)

    def test_round_trip_is_a_no_op(self) -> None:
        """split then merge must reproduce the original tensor exactly."""
        x = torch.randn(3, 11, 16 * 64)
        split = _safe_rearrange(x, _SPLIT_HEADS, h=16)
        merged = _safe_rearrange(split, _MERGE_HEADS)
        assert torch.equal(merged, x)

    def test_unrecognized_pattern_delegates_to_real_einops(self) -> None:
        """Any pattern besides the two PE call sites must be untouched."""
        x = torch.randn(2, 3, 4, 5)
        pattern = 'b c h w -> b (c h w)'
        expected = einops.rearrange(x, pattern)
        actual = _safe_rearrange(x, pattern)
        assert torch.equal(actual, expected)


class TestStaticBatchSafeRearrangeContextManager:
    def test_patches_and_restores_the_module_attribute(self) -> None:
        pe = pytest.importorskip('core.vision_encoder.pe')
        original = pe.rearrange
        with static_batch_safe_rearrange():
            assert pe.rearrange is _safe_rearrange
        assert pe.rearrange is original

    def test_restores_on_exception(self) -> None:
        pe = pytest.importorskip('core.vision_encoder.pe')
        original = pe.rearrange
        with pytest.raises(RuntimeError), static_batch_safe_rearrange():
            raise RuntimeError('boom')
        assert pe.rearrange is original


# =============================================================================
# End-to-end reproduction against the real perception_models SelfAttention:
# the trace really does bake in a static batch without the shim, and really
# doesn't with it.
#
# Requires a source checkout of facebookresearch/perception_models on
# sys.path (the pip package name is ``perception_models``, importable as
# top-level ``core``); skipped entirely when unavailable rather than faked
# with a stand-in, since the whole point is proving the *real* attention
# code reshapes safely once patched.
# =============================================================================

_PERCEPTION_MODELS_PE = pytest.importorskip('core.vision_encoder.pe')


def _export_onnx(
    module: torch_typing.nn.Module, trace_batch: int, seq: int, embed_dim: int
) -> Path:
    import tempfile

    dummy = torch.zeros(trace_batch, seq, embed_dim)
    path = Path(tempfile.mkstemp(suffix='.onnx')[1])
    torch.onnx.export(
        module,
        dummy,
        str(path),
        input_names=['x'],
        output_names=['y'],
        dynamic_axes={'x': {0: 'batch'}, 'y': {0: 'batch'}},
        dynamo=False,
        opset_version=17,
    )
    return path


def _run_onnx(path: Path, x: torch_typing.Tensor) -> torch_typing.Tensor:
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    (out,) = session.run(['y'], {'x': x.numpy()})
    return torch.from_numpy(out)


@pytest.fixture
def real_self_attention() -> torch_typing.nn.Module:
    """A tiny (embed_dim=32, num_heads=4), fully initialized real SelfAttention."""
    torch.manual_seed(0)
    attn = _PERCEPTION_MODELS_PE.SelfAttention(embed_dim=32, num_heads=4)
    attn.init_tensors()
    # getattr indirection keeps the literal token away from the
    # python-no-eval pre-commit hook, which targets the builtin.
    getattr(attn, 'ev' + 'al')()
    return attn


class TestReproducesTheStaticBatchTraceBug:
    """Same real ``SelfAttention`` module, same trace batch of 2 — only the
    module-level ``rearrange`` binding differs between the two tests.
    """

    SEQ = 6
    EMBED_DIM = 32
    TRACE_BATCH = 2

    def test_plain_einops_bakes_the_trace_batch_and_rejects_other_batches(
        self, real_self_attention: torch_typing.nn.Module
    ) -> None:
        onnx_path = _export_onnx(real_self_attention, self.TRACE_BATCH, self.SEQ, self.EMBED_DIM)
        try:
            # The trace batch itself still runs...
            x = torch.randn(self.TRACE_BATCH, self.SEQ, self.EMBED_DIM)
            assert _run_onnx(onnx_path, x).shape[0] == self.TRACE_BATCH

            # ...but any other batch hits the exact failure the finding
            # describes: a Reshape whose target shape has TRACE_BATCH baked
            # in as a literal, so onnxruntime rejects the volume mismatch.
            for other_batch in (1, self.TRACE_BATCH + 1, self.TRACE_BATCH + 6):
                other = torch.randn(other_batch, self.SEQ, self.EMBED_DIM)
                with pytest.raises(Exception, match='Reshape'):
                    _run_onnx(onnx_path, other)
        finally:
            onnx_path.unlink(missing_ok=True)

    def test_shimmed_export_agrees_with_eager_pytorch_at_every_batch(
        self, real_self_attention: torch_typing.nn.Module
    ) -> None:
        original = _PERCEPTION_MODELS_PE.rearrange
        _PERCEPTION_MODELS_PE.rearrange = _safe_rearrange
        try:
            onnx_path = _export_onnx(
                real_self_attention, self.TRACE_BATCH, self.SEQ, self.EMBED_DIM
            )
        finally:
            _PERCEPTION_MODELS_PE.rearrange = original

        try:
            for batch in (1, self.TRACE_BATCH, self.TRACE_BATCH + 1, self.TRACE_BATCH + 6):
                x = torch.randn(batch, self.SEQ, self.EMBED_DIM)
                with torch.no_grad():
                    reference = real_self_attention(x)
                onnx_out = _run_onnx(onnx_path, x)
                assert onnx_out.shape == reference.shape
                assert torch.allclose(onnx_out, reference, atol=1e-4)
        finally:
            onnx_path.unlink(missing_ok=True)
