"""Parity gate (ONNX Runtime vs the reference) and latency benchmark for the PE text encoder."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pe_text_constants import (
    BENCH_PROMPTS,
    INPUT_TENSOR,
    OUTPUT_TENSOR,
    PARITY_BATCH_SIZES,
    PARITY_MIN_COSINE,
    PARITY_PROMPTS,
)


if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)


def trim_to_eot(tokens: Any) -> Any:
    """Drop the all-padding tail after the batch's last EOT position.

    Mirrors ``src.clients.pe_text_backends.trim_text_tokens`` (the exporter is a
    standalone script and does not import ``src``).
    """
    import numpy as np

    arr = np.asarray(tokens, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return arr
    length = int(arr.argmax(axis=1).max()) + 1
    return np.ascontiguousarray(arr[:, :length])


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
