"""ONNX validation of the exported PE image graph and its check against the client contract."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pe_image_constants import IMAGE_SIZE, TRACE_BATCH
from pe_onnx_common import _is_dynamic


if TYPE_CHECKING:
    from pathlib import Path

    from pe_image_triton_config import PETritonConfig


logger = logging.getLogger(__name__)


@dataclass
class OnnxReport:
    """What :func:`validate_onnx` learned about the exported graph."""

    input_name: str | None = None
    input_shape: list[Any] = field(default_factory=list)
    outputs: list[tuple[str, list[Any]]] = field(default_factory=list)
    embedding_name: str | None = None
    embedding_dim: int | None = None
    dynamic_batch: bool | None = None
    skipped: bool = False


def validate_onnx(onnx_path: Path, image_size: int = IMAGE_SIZE) -> OnnxReport:
    """Probe the exported graph with ONNX Runtime on CPU.

    Reports the I/O names/shapes, the detected pooled-embedding output and
    whether the leading axis is dynamic. Returns a report with
    ``skipped=True`` (rather than raising) when onnxruntime isn't installed
    — the export itself is still usable.
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
    report.input_shape = list(first_input.shape)
    report.outputs = [(o.name, list(o.shape)) for o in session.get_outputs()]
    logger.info(f'ONNX inputs:  {[(i.name, i.shape) for i in session.get_inputs()]}')
    logger.info(f'ONNX outputs: {report.outputs}')

    # Concretize the declared input shape so we can run a real forward.
    concrete: list[int] = []
    for axis, dim in enumerate(report.input_shape):
        if _is_dynamic(dim):
            concrete.append({0: TRACE_BATCH, 1: 3}.get(axis, image_size))
        else:
            concrete.append(int(dim))
    logger.info(f'Probing forward with {report.input_name}={concrete}')
    arrays = session.run(None, {report.input_name: np.zeros(concrete, dtype=np.float32)})

    for (name, _declared), array in zip(report.outputs, arrays, strict=False):
        if array.ndim == 2:
            report.embedding_name = name
            report.embedding_dim = int(array.shape[-1])
            report.dynamic_batch = int(array.shape[0]) == concrete[0]
            logger.info(f'Pooled embedding: name={name} dim={report.embedding_dim}')
            break

    # A statically-shaped leading axis is the failure that silently breaks
    # Triton dynamic_batching: the model loads but rejects every batch > 1.
    declared_out = dict(report.outputs).get(report.embedding_name or '', [])
    if declared_out and not _is_dynamic(declared_out[0]):
        report.dynamic_batch = False
    return report


def check_client_contract(report: OnnxReport, cfg: PETritonConfig) -> list[str]:
    """Compare the exported graph against what ``PEEncoder`` will send/read.

    Returns a list of human-readable problems; empty means the graph can be
    served under the rendered config.pbtxt as-is.
    """
    if report.skipped:
        return []

    problems: list[str] = []
    if report.input_name != cfg.input_name:
        problems.append(
            f'input tensor is {report.input_name!r}, but PEEncoder sends {cfg.input_name!r}'
        )
    if report.embedding_name is None:
        problems.append('no 2-D pooled-embedding output found in the exported graph')
    elif report.embedding_name != cfg.output_name:
        problems.append(
            f'embedding output is {report.embedding_name!r}, but PEEncoder reads '
            f'{cfg.output_name!r}'
        )
    if report.embedding_dim is not None and report.embedding_dim != cfg.embedding_dim:
        problems.append(
            f'embedding dim is {report.embedding_dim}, but the curation index expects '
            f'{cfg.embedding_dim}'
        )
    if report.dynamic_batch is False:
        problems.append(
            'the leading axis is static — Triton dynamic_batching will reject batches > 1. '
            f'Re-export with a batch-{TRACE_BATCH} trace dummy.'
        )
    return problems
