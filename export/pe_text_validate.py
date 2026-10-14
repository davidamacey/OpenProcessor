"""ONNX validation of the exported PE text graph and its check against the client contract."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pe_onnx_common import _is_dynamic
from pe_text_constants import CONTEXT_LENGTH, EOT_TOKEN_ID, SOT_TOKEN_ID, TRACE_BATCH


if TYPE_CHECKING:
    from pathlib import Path

    from pe_text_triton_config import PETextTritonConfig


logger = logging.getLogger(__name__)


@dataclass
class OnnxReport:
    """What :func:`validate_onnx` learned about the exported graph."""

    input_name: str | None = None
    input_dtype: str | None = None
    input_shape: list[Any] = field(default_factory=list)
    outputs: list[tuple[str, list[Any]]] = field(default_factory=list)
    embedding_name: str | None = None
    embedding_dim: int | None = None
    dynamic_batch: bool | None = None
    dynamic_tokens: bool | None = None
    skipped: bool = False


def validate_onnx(onnx_path: Path, context_length: int = CONTEXT_LENGTH) -> OnnxReport:
    """Probe the graph with ONNX Runtime on CPU (I/O names, dtype, dynamic batch).

    Returns ``skipped=True`` instead of raising when onnxruntime is missing.
    """
    report = OnnxReport()
    try:
        import numpy as np
        import onnxruntime as ort
    except ModuleNotFoundError:
        logger.warning('onnxruntime not installed — skipping ONNX validation.')
        report.skipped = True
        return report

    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    first_input = session.get_inputs()[0]
    report.input_name = first_input.name
    report.input_dtype = first_input.type
    report.input_shape = list(first_input.shape)
    report.outputs = [(o.name, list(o.shape)) for o in session.get_outputs()]
    logger.info(f'ONNX inputs:  {[(i.name, i.type, i.shape) for i in session.get_inputs()]}')
    logger.info(f'ONNX outputs: {report.outputs}')

    # Probe at a batch different from the trace batch so a baked-in leading
    # dimension fails loudly here rather than in production.
    probe_batch = TRACE_BATCH + 1
    concrete = [
        probe_batch if axis == 0 else (context_length if _is_dynamic(dim) else int(dim))
        for axis, dim in enumerate(report.input_shape)
    ]
    tokens = np.zeros(concrete, dtype=np.int64)
    # SOT ... EOT pattern so argmax pooling picks a real position.
    if tokens.ndim == 2 and tokens.shape[1] >= 2:
        tokens[:, 0], tokens[:, 1] = SOT_TOKEN_ID, EOT_TOKEN_ID
    try:
        arrays = session.run(None, {report.input_name: tokens})
    except Exception as exc:  # ORT raises its own Fail/InvalidArgument types
        logger.error(f'Forward at batch {probe_batch} failed: {exc}')
        report.dynamic_batch = False
        return report

    # The client trims padding after the last EOT; a graph with a baked-in
    # 32-token axis still works, but only untrimmed (slower).
    if tokens.ndim == 2 and tokens.shape[1] > 2:
        try:
            session.run(None, {report.input_name: np.ascontiguousarray(tokens[:, :2])})
            report.dynamic_tokens = True
        except Exception:
            report.dynamic_tokens = False

    for (name, _declared), array in zip(report.outputs, arrays, strict=False):
        if array.ndim == 2:
            report.embedding_name = name
            report.embedding_dim = int(array.shape[-1])
            report.dynamic_batch = int(array.shape[0]) == probe_batch
            break

    declared_out = dict(report.outputs).get(report.embedding_name or '', [])
    if declared_out and not _is_dynamic(declared_out[0]):
        report.dynamic_batch = False
    return report


def check_client_contract(report: OnnxReport, cfg: PETextTritonConfig) -> list[str]:
    """Compare the graph against what ``PEEncoder`` sends/reads. Empty = OK."""
    if report.skipped:
        return []
    problems: list[str] = []
    if report.input_name != cfg.input_name:
        problems.append(
            f'input tensor is {report.input_name!r}, but PEEncoder sends {cfg.input_name!r}'
        )
    if report.input_dtype is not None and report.input_dtype != 'tensor(int64)':
        problems.append(f'input dtype is {report.input_dtype}, but PEEncoder sends int64 token ids')
    token_axis = report.input_shape[1] if len(report.input_shape) == 2 else None
    if (
        token_axis is not None
        and not _is_dynamic(token_axis)
        and int(token_axis) != cfg.context_length
    ):
        problems.append(
            f'token axis is {token_axis}, but the PE tokenizer emits {cfg.context_length} ids'
        )
    if report.dynamic_tokens is False:
        problems.append(
            'the token axis is static — PEEncoder trims padding after the last EOT and '
            'would be rejected. Re-export with a dynamic token axis.'
        )
    if report.embedding_name is None:
        problems.append('no 2-D embedding output found in the exported graph')
    elif report.embedding_name != cfg.output_name:
        problems.append(
            f'embedding output is {report.embedding_name!r}, but PEEncoder reads '
            f'{cfg.output_name!r}'
        )
    if report.embedding_dim is not None and report.embedding_dim != cfg.embedding_dim:
        problems.append(
            f'embedding dim is {report.embedding_dim}, but the image-side index uses '
            f'{cfg.embedding_dim}'
        )
    if report.dynamic_batch is False:
        problems.append(
            'the leading axis is static — batches > 1 will be rejected. '
            f'Re-export with a batch-{TRACE_BATCH} trace dummy.'
        )
    return problems
