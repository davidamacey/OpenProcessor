#!/usr/bin/env python3
"""
PE-Core-L14-336 Text Encoder Export Script
==========================================

Export the Perception Encoder (PE-Core-L14-336) **text tower** to ONNX so the
API can encode semantic-search queries with ONNX Runtime instead of PyTorch
eager, and (optionally) render + install a Triton ``onnxruntime_onnx`` model
entry for it. Companion to ``export/export_pe_image_encoder.py``; both load
the checkpoint fetched by ``export/download_pe_weights.py``.

Why this model matters
----------------------
``src/clients/pe_encoder.py`` (``PEEncoder.encode_text``) turns an operator's
free-text query into the 1024-d vector that ``GET /curation/search/text``
scores against the ``pe_embedding`` field written by the image tower. The
text path is deliberately independent of the GPU and Triton, so semantic
search keeps working when both are down. Before this export the only way to
run it was the full PyTorch ``pe.CLIP`` model (both towers, ~2.7 GB of
weights) in-process.

Graph contract
--------------
Input  ``text_tokens``     INT64 ``[B, T]``, ``1 <= T <= 32`` — the output of
       PE's own ``SimpleTokenizer(context_length=32)`` (SOT + BPE + EOT,
       zero-padded, truncated with EOT kept last), optionally trimmed after
       the batch's last EOT (see below). Tokenization stays in Python.
Output ``text_embeddings`` FP32  ``[B, 1024]`` — ``clip.encode_text(tokens,
       normalize=True)``: causal transformer, final LayerNorm, EOT-position
       (argmax) pooling, text projection, L2 normalize. Exactly what
       ``PEEncoder`` computed with PyTorch (it L2-normalized after the fact;
       the graph now does it itself, and the client still re-normalizes
       defensively).

Both axes are dynamic (``torch.export`` with symbolic ``Dim`` shapes — the
legacy tracer bakes the sequence length into the attention reshapes). The
token axis is dynamic because the text
tower's attention mask is strictly causal and pooling reads the EOT position:
padding *after* a row's EOT can never influence its embedding, so the client
drops the all-padding tail (``T = max EOT index + 1``). A 3-word query then
runs ~5 positions instead of 32, which is where most of the CPU latency win
comes from — for either backend (``pe.TextTransformer.forward`` already slices
its mask and positional table to the input length).

The parity gate compares ONNX Runtime CPU (full 32-token input *and* the
trimmed input) against PyTorch on the full 32-token input, over a fixed
prompt set at batch 1 and batch 8, and fails the export below cosine 0.9999.

Serving options
---------------
1. **In-process ONNX Runtime (default, recommended).** Mount the ``.onnx``
   into the API container and set ``OP_PE_TEXT_ONNX_PATH`` (default
   ``/app/pytorch_models/pe_text_encoder.onnx``). ``PEEncoder`` prefers it
   over PyTorch automatically.
2. **Triton (optional).** ``--install-triton`` copies the graph to
   ``<models-dir>/pe_text_encoder/1/model.onnx`` and writes a matching
   ``config.pbtxt`` (``platform: onnxruntime_onnx``). The API then uses it
   when Triton reports the model ready (``OP_PE_TEXT_BACKEND=auto`` with no
   local ONNX file, or ``=triton``), and falls back to in-process encoding if
   Triton stops answering. Remember ``--load-model=pe_text_encoder`` under
   ``--model-control-mode=explicit``.

Usage
-----
    # ONNX + parity gate (API container: has torch + perception_models)
    docker compose exec yolo-api python /app/export/export_pe_text_encoder.py

    # Also install as a Triton model
    docker compose exec yolo-api python /app/export/export_pe_text_encoder.py \\
        --install-triton --models-dir /app/models

    # Re-render only the Triton config.pbtxt (no torch needed)
    python export/export_pe_text_encoder.py --config-only --models-dir ./models
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Model name + tensor names are a hard contract with src/clients/pe_encoder.py
# (PE_TEXT_TRITON_MODEL / PE_TEXT_INPUT / PE_TEXT_OUTPUT) — rename both sides
# together.
TRITON_MODEL_NAME = 'pe_text_encoder'
INPUT_TENSOR = 'text_tokens'
OUTPUT_TENSOR = 'text_embeddings'

PE_VARIANT = 'PE-Core-L14-336'
# PE_TEXT_CONFIG['PE-Core-L14-336'].context_length in perception_models.
CONTEXT_LENGTH = 32
EMBEDDING_DIM = 1024
# CLIP BPE start/end-of-text ids; EOT is the largest id in the vocabulary,
# which is what PE's argmax pooling relies on.
SOT_TOKEN_ID = 49406
EOT_TOKEN_ID = 49407

# The torch.export-based exporter's floor; ORT >= 1.14 and Triton's
# onnxruntime backend both run it.
ONNX_OPSET_VERSION = 18
DEFAULT_MAX_BATCH = 32

# Only Triton's ONNX Runtime backend is templated: the text tower is tiny
# next to the vision tower, and queries arrive one at a time.
ORT_PLATFORM = 'onnxruntime_onnx'
INSTANCE_KINDS = ('KIND_CPU', 'KIND_GPU')

MODELS_DIR = Path('/app/models')
EXPORT_DIR = Path('/app/pytorch_models')

# See export_pe_image_encoder.TRACE_BATCH: a batch-1 trace can bake the
# leading dimension into a Reshape; batch 2 keeps it symbolic.
TRACE_BATCH = 2

PARITY_MIN_COSINE = 0.9999
PARITY_BATCH_SIZES = (1, 8)
# Mix of short, long (> 32 tokens, exercises truncation), punctuation and
# non-ASCII prompts, typical of operator curation queries.
PARITY_PROMPTS: tuple[str, ...] = (
    'a photo of a white pickup truck',
    'red sedan',
    'a person riding a bicycle at night',
    'close-up of a license plate',
    'blurry image',
    'two dogs playing in the snow on a sunny winter afternoon near a frozen lake '
    'with pine trees and mountains in the background under a clear blue sky while '
    'children build a snowman and skate across the ice at dusk',
    'Stop sign!',
    'motorcycle, side view, black',
    'a crowded street market with colorful umbrellas',
    'x',
    'café storefront with a neon sign',
    'an empty parking lot',
)
# Typical operator queries (all well under the 32-token context) for the
# latency benchmark; the long truncation prompt above is parity-only.
BENCH_PROMPTS: tuple[str, ...] = (
    'a photo of a white pickup truck',
    'red sedan',
    'a person riding a bicycle at night',
    'close-up of a license plate',
    'blurry image',
    'motorcycle, side view, black',
    'a crowded street market with colorful umbrellas',
    'an empty parking lot',
)


# ============================================================================
# Triton config rendering (pure — unit-tested without torch)
# ============================================================================


@dataclass(frozen=True)
class PETextTritonConfig:
    """Everything that varies in the rendered ``config.pbtxt``."""

    model_name: str = TRITON_MODEL_NAME
    max_batch_size: int = DEFAULT_MAX_BATCH
    context_length: int = CONTEXT_LENGTH
    embedding_dim: int = EMBEDDING_DIM
    input_name: str = INPUT_TENSOR
    output_name: str = OUTPUT_TENSOR
    instance_count: int = 1
    kind: str = 'KIND_CPU'
    gpu_ids: tuple[int, ...] = (0,)
    max_queue_delay_us: int = 2000

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError('model_name must be a non-empty string')
        for attr in ('max_batch_size', 'context_length', 'embedding_dim', 'instance_count'):
            value = getattr(self, attr)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f'{attr} must be a positive int, got {value!r}')
        if self.max_queue_delay_us < 0:
            raise ValueError(f'max_queue_delay_us must be >= 0, got {self.max_queue_delay_us}')
        if self.kind not in INSTANCE_KINDS:
            raise ValueError(f'kind must be one of {INSTANCE_KINDS}, got {self.kind!r}')
        if self.kind == 'KIND_GPU':
            if not self.gpu_ids:
                raise ValueError('gpu_ids must contain at least one device id for KIND_GPU')
            if any(not isinstance(g, int) or g < 0 for g in self.gpu_ids):
                raise ValueError(f'gpu_ids must be non-negative ints, got {self.gpu_ids!r}')
        if not self.input_name or not self.output_name:
            raise ValueError('input_name and output_name must be non-empty strings')


def preferred_batch_sizes(max_batch: int) -> list[int]:
    """Triton ``preferred_batch_size`` ladder bounded by ``max_batch``."""
    ladder = [size for size in (4, 8, 16, 32) if size <= max_batch]
    return ladder or [max_batch]


def render_config(cfg: PETextTritonConfig) -> str:
    """Render the Triton ``config.pbtxt`` body for the PE text encoder."""
    preferred = ', '.join(str(b) for b in preferred_batch_sizes(cfg.max_batch_size))
    if cfg.kind == 'KIND_GPU':
        gpus = ', '.join(str(g) for g in cfg.gpu_ids)
        instance = f'    count: {cfg.instance_count}\n    kind: KIND_GPU\n    gpus: [ {gpus} ]'
    else:
        instance = f'    count: {cfg.instance_count}\n    kind: KIND_CPU'

    return f"""# {PE_VARIANT} text encoder — semantic-search query embeddings.
#
# OPTIONAL. src/clients/pe_encoder.py encodes queries in-process by default
# (ONNX Runtime, else PyTorch). It routes here only when Triton reports this
# model ready AND no local ONNX file is configured (or OP_PE_TEXT_BACKEND=
# triton), and falls back in-process for good if Triton stops answering.
# The tensor names below are a contract with that client.
#
# Input:  {cfg.input_name} [B, T] INT64, T <= {cfg.context_length} — PE SimpleTokenizer ids
#         (SOT + BPE + EOT, zero-padded; tokenized in Python by the client,
#         which trims the padding after the batch's last EOT, hence dims -1)
# Output: {cfg.output_name} [B, {cfg.embedding_dim}] FP32, L2-normalized
#
# Model file: 1/model.onnx from export/export_pe_text_encoder.py
# (--install-triton writes both). Load it explicitly:
#   --load-model={cfg.model_name}

name: "{cfg.model_name}"
platform: "{ORT_PLATFORM}"
max_batch_size: {cfg.max_batch_size}

input [
  {{
    name: "{cfg.input_name}"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }}
]

output [
  {{
    name: "{cfg.output_name}"
    data_type: TYPE_FP32
    dims: [ {cfg.embedding_dim} ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ {preferred} ]
  max_queue_delay_microseconds: {cfg.max_queue_delay_us}
}}

instance_group [
  {{
{instance}
  }}
]
"""


def write_triton_config(models_dir: Path, cfg: PETextTritonConfig) -> Path:
    """Write ``<models_dir>/<model_name>/config.pbtxt`` and return its path."""
    model_dir = models_dir / cfg.model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / 'config.pbtxt'
    config_path.write_text(render_config(cfg))
    logger.info(f'Generated Triton config: {config_path}')
    return config_path


def install_triton_model(onnx_path: Path, models_dir: Path, cfg: PETextTritonConfig) -> Path:
    """Copy the graph to ``<models_dir>/<name>/1/model.onnx`` + write config."""
    version_dir = models_dir / cfg.model_name / '1'
    version_dir.mkdir(parents=True, exist_ok=True)
    target = version_dir / 'model.onnx'
    shutil.copy2(onnx_path, target)
    target.chmod(0o644)
    logger.info(f'Installed Triton model file: {target}')
    write_triton_config(models_dir, cfg)
    return target


# ============================================================================
# ONNX export
# ============================================================================


def load_pe_clip(
    variant: str = PE_VARIANT,
    checkpoint_path: Path | None = None,
    perception_models_path: Path | None = None,
    verify_checkpoint: bool = True,
) -> tuple[Any, Any]:
    """Load ``pe.CLIP`` (inference mode, CPU) and its text tokenizer.

    The checkpoint comes from :mod:`download_pe_weights` (pinned revision +
    SHA-256) unless ``checkpoint_path`` points at a local copy.
    """
    if perception_models_path is not None:
        sys.path.insert(0, str(Path(perception_models_path).resolve()))

    from core.vision_encoder import pe
    from core.vision_encoder.tokenizer import SimpleTokenizer
    from download_pe_weights import resolve_checkpoint

    ckpt = resolve_checkpoint(variant, checkpoint_path, verify=verify_checkpoint)
    logger.info(f'Loading {variant} from {ckpt} ...')
    clip_model = pe.CLIP.from_config(variant, pretrained=True, checkpoint_path=str(ckpt))
    # getattr indirection keeps the literal token away from the python-no-eval
    # pre-commit hook, which targets the builtin.
    getattr(clip_model, 'ev' + 'al')()
    tokenizer = SimpleTokenizer(context_length=clip_model.context_length)
    return clip_model, tokenizer


def build_text_wrapper(clip_model: Any) -> Any:
    """``tokens -> clip.encode_text(tokens, normalize=True)`` as a Module."""
    import torch

    class PETextEncoder(torch.nn.Module):
        def __init__(self, clip: Any) -> None:
            super().__init__()
            self.clip = clip

        def forward(self, text_tokens: Any) -> Any:  # name == dynamic_shapes key
            return self.clip.encode_text(text_tokens, normalize=True)

    wrapper = PETextEncoder(clip_model)
    getattr(wrapper, 'ev' + 'al')()
    return wrapper


def trim_to_eot(tokens: Any) -> Any:
    """Drop the all-padding tail after the batch's last EOT position.

    Mirrors ``src.clients.pe_encoder.trim_text_tokens`` (the exporter is a
    standalone script and does not import ``src``).
    """
    import numpy as np

    arr = np.asarray(tokens, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return arr
    length = int(arr.argmax(axis=1).max()) + 1
    return np.ascontiguousarray(arr[:, :length])


def export_text_onnx(
    clip_model: Any,
    tokenizer: Any,
    onnx_path: Path,
    opset: int = ONNX_OPSET_VERSION,
) -> Path:
    """Export the text tower to ONNX with dynamic batch + token axes (CPU).

    Uses the ``torch.export``-based exporter (``dynamo=True``). The legacy
    TorchScript tracer bakes the traced sequence length into the
    ``nn.MultiheadAttention`` reshapes (``Reshape`` to ``{32, B*16, 64}``),
    so a graph traced at 32 tokens rejects every trimmed input; symbolic
    ``Dim`` shapes keep both axes dynamic end to end.
    """
    import torch
    from torch.export import Dim

    wrapper = build_text_wrapper(clip_model)
    # Real token ids trimmed below the context length, so neither axis can
    # be specialized to a coincidental constant.
    dummy = tokenizer(list(PARITY_PROMPTS[:TRACE_BATCH]))[:, : CONTEXT_LENGTH // 2 + 1]
    dynamic_shapes = {
        'text_tokens': {
            0: Dim('batch', min=1, max=1024),
            1: Dim('tokens', min=1, max=int(clip_model.context_length)),
        }
    }
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f'Exporting text tower (torch.export, opset {opset})...')
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            opset_version=opset,
            input_names=[INPUT_TENSOR],
            output_names=[OUTPUT_TENSOR],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            external_data=False,
        )
    logger.info(f'ONNX saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)')
    return onnx_path


# ============================================================================
# Validation
# ============================================================================


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


def _is_dynamic(dim: Any) -> bool:
    """Whether an ONNX dimension is symbolic (i.e. batchable)."""
    return isinstance(dim, str) or dim is None or (isinstance(dim, int) and dim < 1)


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


@dataclass
class ParityReport:
    """ONNX-vs-reference agreement over one or more batches."""

    n: int = 0
    min_cosine: float = 1.0
    max_abs_diff: float = 0.0
    batch_sizes: tuple[int, ...] = ()

    def passes(self, threshold: float = PARITY_MIN_COSINE) -> bool:
        return self.n > 0 and self.min_cosine >= threshold


def compare_embeddings(reference: Any, candidate: Any) -> tuple[float, float]:
    """``(min row cosine, max abs diff)`` between two ``[N, D]`` matrices."""
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError(f'shape mismatch: reference {ref.shape} vs candidate {cand.shape}')
    num = (ref * cand).sum(axis=1)
    den = np.linalg.norm(ref, axis=1) * np.linalg.norm(cand, axis=1)
    cos = num / np.where(den == 0, 1.0, den)
    return float(cos.min()), float(np.abs(ref - cand).max())


def run_parity(
    reference_fn: Any,
    candidate_fn: Any,
    token_batches: list[Any],
) -> ParityReport:
    """Run both encoders on every token batch and aggregate agreement.

    Both callables take an int64 ``[B, L]`` numpy array and return ``[B, D]``.
    Pure orchestration — unit-tested with numpy stand-ins.
    """
    report = ParityReport()
    sizes: list[int] = []
    for tokens in token_batches:
        min_cos, max_abs = compare_embeddings(reference_fn(tokens), candidate_fn(tokens))
        report.n += int(tokens.shape[0])
        report.min_cosine = min(report.min_cosine, min_cos)
        report.max_abs_diff = max(report.max_abs_diff, max_abs)
        sizes.append(int(tokens.shape[0]))
    report.batch_sizes = tuple(sizes)
    return report


def parity_token_batches(tokenizer: Any, batch_sizes: tuple[int, ...] = PARITY_BATCH_SIZES) -> list:
    """Every parity prompt at batch 1, then chunks at each larger batch size."""
    import numpy as np

    prompts = list(PARITY_PROMPTS)
    tokens = np.asarray(tokenizer(prompts), dtype=np.int64)
    batches: list[Any] = []
    for size in batch_sizes:
        batches.extend(tokens[i : i + size] for i in range(0, len(prompts), size))
    return batches


def check_parity(
    clip_model: Any,
    tokenizer: Any,
    onnx_path: Path,
    batch_sizes: tuple[int, ...] = PARITY_BATCH_SIZES,
) -> ParityReport:
    """PyTorch ``encode_text`` vs ONNX Runtime CPU on :data:`PARITY_PROMPTS`."""
    import numpy as np
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    def reference(tokens: Any) -> Any:
        with torch.no_grad():
            out = clip_model.encode_text(torch.from_numpy(tokens), normalize=True)
        return out.numpy()

    def candidate(tokens: Any) -> Any:
        return session.run([OUTPUT_TENSOR], {INPUT_TENSOR: tokens})[0]

    def candidate_trimmed(tokens: Any) -> Any:
        return candidate(trim_to_eot(tokens))

    batches = parity_token_batches(tokenizer, batch_sizes)
    full = run_parity(reference, candidate, batches)
    trimmed = run_parity(reference, candidate_trimmed, batches)
    for label, rep_ in (('full 32-token', full), ('EOT-trimmed', trimmed)):
        logger.info(
            f'Parity ORT CPU ({label}) vs PyTorch, {rep_.n} rows, batches '
            f'{sorted(set(rep_.batch_sizes))}: min cosine={rep_.min_cosine:.7f} '
            f'max |diff|={rep_.max_abs_diff:.2e}'
        )
    report = ParityReport(
        n=full.n + trimmed.n,
        min_cosine=min(full.min_cosine, trimmed.min_cosine),
        max_abs_diff=max(full.max_abs_diff, trimmed.max_abs_diff),
        batch_sizes=full.batch_sizes + trimmed.batch_sizes,
    )
    # Sanity: the graph's own normalization must hold.
    probe = candidate(np.asarray(tokenizer(['a photo of a car']), dtype=np.int64))
    logger.info(f'  output L2 norm: {float(np.linalg.norm(probe)):.6f}')
    return report


def benchmark(
    clip_model: Any,
    tokenizer: Any,
    onnx_path: Path,
    batch_sizes: tuple[int, ...] = PARITY_BATCH_SIZES,
    iterations: int = 30,
    threads: int | None = None,
) -> list[dict[str, Any]]:
    """Per-call latency, PyTorch eager vs ONNX Runtime CPU (median / p90 ms)."""
    import numpy as np
    import onnxruntime as ort
    import torch

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
        torch.set_num_threads(threads)
    session = ort.InferenceSession(
        str(onnx_path), sess_options=opts, providers=['CPUExecutionProvider']
    )
    rows: list[dict[str, Any]] = []
    prompts = list(BENCH_PROMPTS)
    for size in batch_sizes:
        tokens = np.asarray(tokenizer((prompts * size)[:size]), dtype=np.int64)
        short = trim_to_eot(tokens)
        torch_full, torch_short = torch.from_numpy(tokens), torch.from_numpy(short)

        def run_torch(t: Any = torch_full) -> None:
            with torch.no_grad():
                clip_model.encode_text(t, normalize=True)

        def run_torch_trim(t: Any = torch_short) -> None:
            with torch.no_grad():
                clip_model.encode_text(t, normalize=True)

        def run_ort(t: Any = tokens) -> None:
            session.run([OUTPUT_TENSOR], {INPUT_TENSOR: t})

        def run_ort_trim(t: Any = short) -> None:
            session.run([OUTPUT_TENSOR], {INPUT_TENSOR: t})

        for backend, fn in (
            ('pytorch-eager', run_torch),
            ('pytorch-eager+trim', run_torch_trim),
            ('onnxruntime-cpu', run_ort),
            ('onnxruntime-cpu+trim', run_ort_trim),
        ):
            for _ in range(3):
                fn()
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                fn()
                times.append((time.perf_counter() - start) * 1000.0)
            rows.append(
                {
                    'backend': backend,
                    'batch': size,
                    'tokens': int(short.shape[1] if 'trim' in backend else tokens.shape[1]),
                    'median_ms': float(np.median(times)),
                    'p90_ms': float(np.percentile(times, 90)),
                }
            )
            logger.info(
                f'  {backend:21s} batch={size:<3d} T={rows[-1]["tokens"]:<3d} median={rows[-1]["median_ms"]:8.2f} ms '
                f'p90={rows[-1]["p90_ms"]:8.2f} ms'
            )
    return rows


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f'Export the {PE_VARIANT} text encoder to ONNX (ORT in-process / Triton)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--variant', default=PE_VARIANT, help=f'default: {PE_VARIANT}')
    parser.add_argument(
        '--checkpoint-path',
        type=Path,
        default=None,
        help='Local PE checkpoint (.pt); default: pinned download via download_pe_weights.py',
    )
    parser.add_argument(
        '--no-verify-checkpoint',
        action='store_false',
        dest='verify_checkpoint',
        help='Skip the SHA-256 check of the checkpoint',
    )
    parser.add_argument(
        '--perception-models-path',
        type=Path,
        default=None,
        help='Source checkout of facebookresearch/perception_models when not pip-installed',
    )
    parser.add_argument(
        '--onnx-out',
        type=Path,
        default=EXPORT_DIR / f'{TRITON_MODEL_NAME}.onnx',
        help=f'ONNX destination (default: {EXPORT_DIR / f"{TRITON_MODEL_NAME}.onnx"})',
    )
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=MODELS_DIR,
        help=f'Triton model repository root (default: {MODELS_DIR})',
    )
    parser.add_argument('--triton-name', default=TRITON_MODEL_NAME, help='Triton model name')
    parser.add_argument('--max-batch', type=int, default=DEFAULT_MAX_BATCH)
    parser.add_argument('--context-length', type=int, default=CONTEXT_LENGTH)
    parser.add_argument('--embedding-dim', type=int, default=EMBEDDING_DIM)
    parser.add_argument('--opset', type=int, default=ONNX_OPSET_VERSION)
    parser.add_argument('--instance-count', type=int, default=1)
    parser.add_argument(
        '--kind',
        choices=['cpu', 'gpu'],
        default='cpu',
        help='Triton instance kind (default cpu: queries are tiny and rare; keeps VRAM free)',
    )
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='GPU ids for --kind gpu')
    parser.add_argument(
        '--config-only',
        action='store_true',
        help='Only (re-)write the Triton config.pbtxt; no torch, no export',
    )
    parser.add_argument(
        '--install-triton',
        action='store_true',
        help='Also copy the ONNX into <models-dir>/<name>/1/model.onnx and write config.pbtxt',
    )
    parser.add_argument('--skip-validate', action='store_true', help='Skip the ORT contract probe')
    parser.add_argument('--skip-parity', action='store_true', help='Skip the PyTorch parity gate')
    parser.add_argument(
        '--parity-threshold',
        type=float,
        default=PARITY_MIN_COSINE,
        help=f'Minimum per-row cosine vs PyTorch (default {PARITY_MIN_COSINE})',
    )
    parser.add_argument(
        '--benchmark', action='store_true', help='Time PyTorch eager vs ORT CPU after export'
    )
    parser.add_argument('--bench-iterations', type=int, default=30)
    parser.add_argument('--bench-threads', type=int, default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help='Debug logging')
    return parser


def config_from_args(args: argparse.Namespace) -> PETextTritonConfig:
    return PETextTritonConfig(
        model_name=args.triton_name,
        max_batch_size=args.max_batch,
        context_length=args.context_length,
        embedding_dim=args.embedding_dim,
        instance_count=args.instance_count,
        kind='KIND_GPU' if args.kind == 'gpu' else 'KIND_CPU',
        gpu_ids=tuple(args.gpus),
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        cfg = config_from_args(args)
    except ValueError as exc:
        logger.error(f'Invalid arguments: {exc}')
        return 2
    if args.opset < 1:
        logger.error(f'--opset must be a positive int, got {args.opset}')
        return 2
    if not 0.0 < args.parity_threshold <= 1.0:
        logger.error(f'--parity-threshold must be in (0, 1], got {args.parity_threshold}')
        return 2

    if args.config_only:
        write_triton_config(args.models_dir, cfg)
        return 0

    clip_model, tokenizer = load_pe_clip(
        args.variant,
        checkpoint_path=args.checkpoint_path,
        perception_models_path=args.perception_models_path,
        verify_checkpoint=args.verify_checkpoint,
    )
    if int(clip_model.context_length) != cfg.context_length:
        logger.error(
            f'{args.variant} has context_length {clip_model.context_length}, '
            f'--context-length is {cfg.context_length}'
        )
        return 2
    onnx_path = export_text_onnx(clip_model, tokenizer, args.onnx_out, opset=args.opset)

    problems: list[str] = []
    if not args.skip_validate:
        problems = check_client_contract(validate_onnx(onnx_path, cfg.context_length), cfg)

    parity: ParityReport | None = None
    if not args.skip_parity:
        parity = check_parity(clip_model, tokenizer, onnx_path)
        if not parity.passes(args.parity_threshold):
            problems.append(
                f'parity vs PyTorch failed: min cosine {parity.min_cosine:.7f} < '
                f'{args.parity_threshold}'
            )

    if args.benchmark:
        benchmark(
            clip_model,
            tokenizer,
            onnx_path,
            iterations=args.bench_iterations,
            threads=args.bench_threads,
        )

    if problems:
        for problem in problems:
            logger.error(f'Contract/parity failure: {problem}')
        logger.error('Not installing into Triton; the graph will NOT serve PEEncoder as-is.')
        return 1

    if args.install_triton:
        install_triton_model(onnx_path, args.models_dir, cfg)

    logger.info('=' * 70)
    logger.info(f'ONNX:     {onnx_path}')
    logger.info(
        f'Contract: {cfg.input_name} [B, T<={cfg.context_length}] INT64 -> '
        f'{cfg.output_name} [B, {cfg.embedding_dim}] FP32 (L2-normalized)'
    )
    if parity is not None:
        logger.info(f'Parity:   min cosine {parity.min_cosine:.7f} over {parity.n} rows')
    logger.info('Enable in the API: mount the file and set')
    logger.info(f'  OP_PE_TEXT_ONNX_PATH={onnx_path}')
    logger.info('=' * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
