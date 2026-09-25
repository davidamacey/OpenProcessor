"""Narrow ``einops.rearrange`` shim for exporting PE-Core through ``torch.onnx.export``.

``core.vision_encoder.pe.SelfAttention`` (used by both the image and text
towers — see ``perception_models``' ``core/vision_encoder/pe.py``) reshapes
its q/k/v tensors with two fixed ``einops.rearrange`` patterns:

    q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
    attn = rearrange(attn, "b h s d -> b s (h d)")

Under the legacy TorchScript tracer (``torch.onnx.export(..., dynamo=False)``,
what the image-tower export uses), ``einops.rearrange`` reads the traced
tensor's concrete ``.shape`` to build its internal reshape, which bakes the
traced batch size in as a literal ONNX ``Reshape`` constant (e.g.
``resblocks.*/attn/Reshape_4`` -> ``{2, 577, 16, 64}``). TensorRT then
rejects every batch size other than the one used at trace time
("reshape would change volume"), and this project's own export contract
gate (``check_client_contract`` / ``validate_onnx``) catches the resulting
static leading axis and exits non-zero.

``torch.unflatten`` / ``.transpose`` / ``.flatten`` are ordinary tensor ops
the tracer represents symbolically (their output shape is expressed in terms
of the input's symbolic dimensions, not baked-in constants), so swapping
just these two call sites removes the trap without touching anything else
in ``pe.py``. This shim patches ``core.vision_encoder.pe.rearrange`` (the
name bound into that module's namespace by ``from einops import rearrange``)
to intercept exactly those two literal patterns and fall back to real
``einops.rearrange`` for anything else, so it stays inert if upstream ever
changes unrelated call sites.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Iterator


# The exact two patterns SelfAttention.forward uses (perception_models'
# core/vision_encoder/pe.py). Matched verbatim — anything else falls
# through to real einops so this shim can't silently mask a future upstream
# rearrange call it wasn't written for.
_SPLIT_HEADS = 'b s (h d) -> b h s d'
_MERGE_HEADS = 'b h s d -> b s (h d)'


def _safe_rearrange(tensor: Any, pattern: str, **axes_lengths: int) -> Any:
    """Trace-safe equivalents of the two SelfAttention rearrange patterns."""
    if pattern == _SPLIT_HEADS:
        h = axes_lengths['h']
        # [b, s, h*d] -> [b, s, h, d] -> [b, h, s, d]
        return tensor.unflatten(-1, (h, -1)).transpose(1, 2)
    if pattern == _MERGE_HEADS:
        # [b, h, s, d] -> [b, s, h, d] -> [b, s, h*d]
        return tensor.transpose(1, 2).flatten(2)

    from einops import rearrange as _real_rearrange

    return _real_rearrange(tensor, pattern, **axes_lengths)


@contextlib.contextmanager
def static_batch_safe_rearrange() -> Iterator[None]:
    """Patch ``core.vision_encoder.pe.rearrange`` for the duration of the block.

    Must be entered before the first ``core.vision_encoder.pe.CLIP`` forward
    pass that will be traced (i.e. wrap both model construction and the
    ``torch.onnx.export`` call), and is restored on exit regardless of
    exceptions.
    """
    from core.vision_encoder import pe as pe_module

    original = pe_module.rearrange
    pe_module.rearrange = _safe_rearrange
    try:
        yield
    finally:
        pe_module.rearrange = original
