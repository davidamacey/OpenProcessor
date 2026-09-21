"""SAM 3 model pool + inference internals for the segmenter service.

This module is deliberately free of FastAPI/pydantic: it owns the GPU
side (building the model pool, running a promptable forward, turning raw
model output into normalized boxes) and hands back plain
:class:`Candidate` dataclasses. ``main.py`` owns the HTTP contract and
maps those onto the wire models. Keeping the split means the inference
math can be exercised without standing up an ASGI app, and the HTTP
contract can be exercised without a GPU.

Nothing here is domain-specific: the text prompt is a per-request
argument all the way down, there are no class names, and no dataset
paths. The only thing that is model-specific is the SAM 3 build/forward
itself, which is what this image exists to serve.

Heavy imports (``torch``, ``sam3``) are deliberately function-local so
this module is importable — and testable — in an environment that has
neither.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Sequence

    import numpy as np
    from PIL import Image


log = logging.getLogger('segmenter')


@dataclass(frozen=True, slots=True)
class Candidate:
    """One segmented instance, in the *input image's* normalized frame.

    Attributes:
        bbox_norm: ``(x1, y1, x2, y2)``, each in ``[0, 1]`` relative to
            the submitted image (which is typically a crop of a larger
            source frame — re-projecting to the source frame is the
            caller's job).
        score: Model confidence for this instance.
        mask_iou: Rectangularity (mask area / mask-bbox area), or
            ``None`` when the mask head is disabled. ``~1.0`` means the
            mask fills its bounding box.
    """

    bbox_norm: tuple[float, float, float, float]
    score: float
    mask_iou: float | None = None


def _env_flag(name: str, default: str = '0') -> bool:
    return os.getenv(name, default).strip().lower() in {'1', 'true', 'yes', 'on'}


def device_name() -> str:
    """The torch device string this process hosts the model on."""
    return os.getenv('SEGMENTER_DEVICE', 'cuda:0')


# =============================================================================
# Model construction
# =============================================================================


def _disable_activation_checkpointing(model: Any) -> int:
    """Walk the model and disable activation checkpointing on every module.

    SAM 3's ``build_sam3_image_model`` hard-codes ``use_act_checkpoint=True``
    in the vision backbone, transformer, and segmentation head. Activation
    checkpointing trades compute for memory by re-running forward during
    backward — pointless for inference where there is no backward pass.
    Disabling it cuts ~15-25% off the per-call latency.

    The attribute names vary across SAM 3 modules (``use_act_checkpoint``,
    ``act_ckpt``, ``use_act_ckpt``); set whichever exists.

    Returns:
        How many modules were changed (logged at load time).
    """
    n_disabled = 0
    for module in model.modules():
        for attr in ('use_act_checkpoint', 'act_ckpt', 'use_act_ckpt'):
            if hasattr(module, attr) and getattr(module, attr):
                setattr(module, attr, False)
                n_disabled += 1
    return n_disabled


def _patch_processor_for_no_masks(processor: Any) -> None:
    """Replace ``Sam3Processor._forward_grounding`` to skip ``pred_masks``.

    The upstream ``_forward_grounding`` unconditionally reads
    ``outputs['pred_masks']`` and interpolates it to image size. When the
    model is built with ``enable_segmentation=False`` it does not produce
    ``pred_masks`` at all, so the processor raises ``KeyError``. Boxes +
    scores are enough for the wire contract (``mask_iou`` simply comes
    back ``None``), so a mask-less variant is sufficient. Bound as a
    method on this processor instance only — other instances, including
    ones sharing the same model, are unaffected.
    """
    import torch as _torch
    from sam3.model.box_ops import box_cxcywh_to_xyxy

    @_torch.inference_mode()
    def _forward_grounding_no_masks(self: Any, state: dict) -> dict:
        outputs = self.model.forward_grounding(
            backbone_out=state['backbone_out'],
            find_input=self.find_stage,
            geometric_prompt=state['geometric_prompt'],
            find_target=None,
        )
        out_bbox = outputs['pred_boxes']
        out_logits = outputs['pred_logits']
        out_probs = out_logits.sigmoid()
        presence_score = outputs['presence_logit_dec'].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_bbox = out_bbox[keep]

        boxes = box_cxcywh_to_xyxy(out_bbox)
        img_h = state['original_height']
        img_w = state['original_width']
        scale_fct = _torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        state['boxes'] = boxes
        state['scores'] = out_probs
        state['masks'] = None
        state['masks_logits'] = None
        return state

    processor._forward_grounding = types.MethodType(_forward_grounding_no_masks, processor)


def load_one_instance(device: str) -> tuple[Any, Any]:
    """Build a single SAM 3 model + processor on ``device``.

    Production tunings applied at load:

    * Weights stay fp32; the autocast wrapper inside :func:`segment_images`
      runs every op in bf16 — this matches ``Sam3BasePredictor.add_prompt``
      (the official inference path). Casting weights to bf16 directly
      breaks the fp32 input path through ``patch_embed``.
    * Activation checkpointing is disabled on every module — it is a
      training-time memory optimization that only adds inference overhead.
    * ``SEGMENTER_COMPILE=1`` opts into ``torch.compile`` via the upstream
      ``build_sam3_image_model(compile=True)`` flag, which wires
      ``compile_mode='default'`` through to the vision backbone,
      segmentation head and pixel decoder. The first call pays a 30-60 s
      warmup; steady-state throughput improves ~25-50%.

    Returns:
        ``(model, processor)``. Callers that only need the processor can
        discard the model reference — the processor holds it.
    """
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    use_compile = _env_flag('SEGMENTER_COMPILE', '0')
    enable_masks = _env_flag('SEGMENTER_ENABLE_MASKS', '1')
    log.info(
        'building sam3 image model on device=%s (compile=%s, masks=%s)',
        device,
        use_compile,
        enable_masks,
    )
    t0 = time.perf_counter()
    model = build_sam3_image_model(compile=use_compile, enable_segmentation=enable_masks)
    if device != 'cpu':
        model = model.to(device)
    n_disabled = _disable_activation_checkpointing(model)
    log.info('disabled act_ckpt on %d modules (inference-only)', n_disabled)
    model.eval()
    processor = Sam3Processor(model)
    if not enable_masks:
        _patch_processor_for_no_masks(processor)
        log.info('patched processor for mask-disabled mode')
    log.info('sam3 instance ready in %.1fs', time.perf_counter() - t0)
    return model, processor


def _log_vram(stage: str) -> None:
    """Log free/total VRAM at a checkpoint so the boot log shows pool cost."""
    try:
        import torch as _torch

        free, total = _torch.cuda.mem_get_info()
        used = total - free
        log.info(
            'segmenter_vram[%s]: used=%.2f GB / total=%.2f GB / free=%.2f GB',
            stage,
            used / 1024**3,
            total / 1024**3,
            free / 1024**3,
        )
    except Exception as exc:
        # A VRAM probe must never break boot.
        log.warning('segmenter_vram probe failed at %s: %s', stage, exc)


def _export_hf_token() -> None:
    """Normalize whichever HuggingFace token var the operator supplied.

    ``huggingface_hub`` reads ``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN``;
    accept the ``HUGGING_TOKEN`` spelling too so an existing deployment's
    ``.env`` keeps working. The weights repo is gated, so a missing token
    is a warning at boot rather than a surprise 401 mid-download.
    """
    token = (
        os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HUGGING_TOKEN')
    )
    if not token:
        log.warning(
            'no HF_TOKEN/HUGGINGFACE_HUB_TOKEN/HUGGING_TOKEN in environment; '
            'weight download will fail if the model repo is gated'
        )
        return
    os.environ.setdefault('HF_TOKEN', token)
    os.environ.setdefault('HUGGINGFACE_HUB_TOKEN', token)


def load_processors() -> list[Any]:
    """Build the SAM 3 processor pool according to the ``SEGMENTER_*`` env.

    Two modes:

    * **Independent weights** (default, ``SEGMENTER_SHARED_WEIGHTS=0``):
      every processor gets its own model copy. Safe but VRAM-heavy — at
      ~4 GB/instance in bf16 a 48 GB card caps out around 10 instances.

    * **Shared weights** (``SEGMENTER_SHARED_WEIGHTS=1``): build the model
      once, wrap it in N processors. ``Sam3Processor.set_image()`` is
      stateless on the processor (it returns an explicit ``state`` dict
      that is carried through the rest of the call) and the model runs in
      eval mode with no parameter writes, so N processors over one model
      is concurrency-safe for inference. CUDA already serializes kernel
      launches on a device, so per-processor parallelism comes from
      overlapping Python pre/post-processing and HTTP overhead rather
      than simultaneous GPU compute. The trade is duplicated weights for
      headroom: more instances, larger activation budgets, or room for
      another model on the same card.

    If shared-mode init fails (upstream API drift, OOM, …) this falls
    back to the independent path rather than failing the boot.
    """
    _export_hf_token()
    device = device_name()
    n_instances = max(1, int(os.getenv('SEGMENTER_INSTANCES', '2')))
    shared = _env_flag('SEGMENTER_SHARED_WEIGHTS', '0')

    _log_vram('pre-load')

    if shared:
        try:
            log.info(
                'loading sam3 pool (shared weights): 1 model, %d processors on %s',
                n_instances,
                device,
            )
            shared_model, _ = load_one_instance(device)
            _log_vram('after-shared-model-load')

            from sam3.model.sam3_image_processor import Sam3Processor

            enable_masks = _env_flag('SEGMENTER_ENABLE_MASKS', '1')
            instances: list[Any] = []
            for i in range(n_instances):
                log.info('wiring processor %d/%d', i + 1, n_instances)
                processor = Sam3Processor(shared_model)
                if not enable_masks:
                    _patch_processor_for_no_masks(processor)
                instances.append(processor)
            _log_vram('after-shared-pool-ready')
        except Exception:
            log.exception(
                'shared-weights mode failed; falling back to an independent-weights '
                'pool with SEGMENTER_INSTANCES=%d',
                n_instances,
            )
            # Don't try to recover GPU state surgically — drop the cache
            # and let the independent path rebuild cleanly.
            with contextlib.suppress(Exception):
                import torch as _torch

                _torch.cuda.empty_cache()
        else:
            return instances

    log.info('loading sam3 pool (independent weights): %d instances on %s', n_instances, device)
    instances = []
    for i in range(n_instances):
        log.info('loading instance %d/%d', i + 1, n_instances)
        _, processor = load_one_instance(device)
        instances.append(processor)
    _log_vram('after-independent-pool-ready')
    return instances


# =============================================================================
# Processor pool
# =============================================================================


class ProcessorPool:
    """N processors, each behind its own lock, handed out round-robin.

    A single ``Sam3Processor`` is not safe to drive concurrently, but N
    of them are — so the pool is what bounds in-flight GPU work. Callers
    use it as an async context manager::

        async with pool.acquire() as processor:
            ...

    The pool is constructed with an explicit processor list rather than
    loading them itself, which keeps :func:`load_processors` (GPU, slow,
    untestable) separate from the scheduling logic (pure, fast, tested).
    """

    __slots__ = ('_locks', '_processors', '_rr_index', '_rr_lock')

    def __init__(self, processors: Iterable[Any]) -> None:
        self._processors: list[Any] = list(processors)
        self._locks: list[asyncio.Lock] = [asyncio.Lock() for _ in self._processors]
        self._rr_index = 0
        self._rr_lock = asyncio.Lock()

    def __len__(self) -> int:
        return len(self._processors)

    def __bool__(self) -> bool:
        return bool(self._processors)

    @contextlib.asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        """Yield the next available processor, waiting only if all are busy.

        Tries each processor once in round-robin order and takes the
        first whose lock is free; if they are all busy, blocks on the
        round-robin-selected one. That keeps fairness without preferring
        any single instance.

        Raises:
            RuntimeError: if the pool is empty (model not loaded yet).
        """
        n = len(self._processors)
        if n == 0:
            raise RuntimeError('processor pool is empty')
        async with self._rr_lock:
            start = self._rr_index
            self._rr_index = (self._rr_index + 1) % n

        idx = start
        for offset in range(n):
            candidate = (start + offset) % n
            if not self._locks[candidate].locked():
                idx = candidate
                break
        await self._locks[idx].acquire()
        try:
            yield self._processors[idx]
        finally:
            self._locks[idx].release()


# =============================================================================
# Inference
# =============================================================================


def _bbox_to_norm(
    box: Any,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Convert a possibly-pixel, possibly-normalized box to normalized.

    SAM 3's processor emits boxes in normalized ``[0, 1]`` for some
    checkpoints and in pixel space for others. Auto-detect via the max
    value (no normalized box can have an axis > 1.0). The result is
    clamped and canonicalized so ``x1 <= x2`` and ``y1 <= y2``.
    """
    import numpy as np

    arr = np.asarray(box, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return (0.0, 0.0, 0.0, 0.0)
    if arr.max() <= 1.0:
        x1, y1, x2, y2 = arr[:4]
    else:
        x1 = arr[0] / max(width, 1)
        y1 = arr[1] / max(height, 1)
        x2 = arr[2] / max(width, 1)
        y2 = arr[3] / max(height, 1)
    x1c = max(0.0, min(1.0, float(min(x1, x2))))
    y1c = max(0.0, min(1.0, float(min(y1, y2))))
    x2c = max(0.0, min(1.0, float(max(x1, x2))))
    y2c = max(0.0, min(1.0, float(max(y1, y2))))
    return (x1c, y1c, x2c, y2c)


def _rectangularity(mask: np.ndarray) -> float:
    """Mask area / mask-bbox area. ``~1.0`` means the mask fills its bbox.

    A soft shape signal for the caller: a near-rectangular target scores
    high, a blob that merely overlaps the box scores low.
    """
    import numpy as np

    if mask.ndim == 3:
        mask = mask[0]
    mask_bin = mask > 0.5
    rows = np.any(mask_bin, axis=1)
    cols = np.any(mask_bin, axis=0)
    if not rows.any() or not cols.any():
        return 0.0
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox_area = float((rmax - rmin + 1) * (cmax - cmin + 1))
    if bbox_area <= 0:
        return 0.0
    return float(mask_bin.sum()) / bbox_area


def _forward_with_cached_prompt(processor: Any, state: dict, prompt: str) -> dict:
    """``Sam3Processor.set_text_prompt`` with a per-processor prompt cache.

    The text tower runs ~20-50 ms per call. Deployments send the same
    prompt for every request in a run, so caching the text-encoder output
    per processor instance reduces that to ~0 after the first request.
    The cache is keyed by the prompt string, so a caller that varies its
    prompt still gets correct (just uncached) results.
    """
    cache = getattr(processor, '_prompt_cache', None)
    if cache is None:
        cache = {}
        processor._prompt_cache = cache

    text_outputs = cache.get(prompt)
    if text_outputs is None:
        text_outputs = processor.model.backbone.forward_text([prompt], device=processor.device)
        cache[prompt] = text_outputs

    state['backbone_out'].update(text_outputs)
    if 'geometric_prompt' not in state:
        state['geometric_prompt'] = processor.model._get_dummy_prompt()
    return processor._forward_grounding(state)


def _autocast_ctx() -> Any:
    """bf16 autocast on CUDA, a no-op elsewhere.

    Weights stay fp32 and every op runs in bf16 under autocast. Without
    it the vitdet MLP layers fail with ``mat1 and mat2 must have the same
    dtype``.
    """
    import torch as _torch

    if _torch.cuda.is_available():
        return _torch.autocast(device_type='cuda', dtype=_torch.bfloat16)
    return _torch.autocast(device_type='cpu', dtype=_torch.bfloat16, enabled=False)


def _candidates_from_output(
    output: dict,
    width: int,
    height: int,
    max_candidates: int,
) -> list[Candidate]:
    """Turn one raw grounding output into sorted, normalized candidates."""
    import numpy as np

    boxes = output.get('boxes', [])
    scores = output.get('scores', [])
    masks = output.get('masks')  # None when the mask head is disabled.

    n = min(
        len(boxes) if hasattr(boxes, '__len__') else 0,
        len(scores) if hasattr(scores, '__len__') else 0,
    )
    candidates: list[Candidate] = []
    for i in range(n):
        box = boxes[i]
        score = scores[i]
        if hasattr(box, 'cpu'):
            box = box.cpu().numpy()
        score = float(score.item()) if hasattr(score, 'item') else float(score)
        bbox_norm = _bbox_to_norm(box, width, height)
        if bbox_norm == (0.0, 0.0, 0.0, 0.0):
            # Degenerate box; skip rather than emit a zero-area candidate.
            continue
        mask_iou: float | None = None
        if masks is not None and len(masks) > i:
            mask = masks[i]
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            mask_iou = _rectangularity(np.asarray(mask))
        candidates.append(Candidate(bbox_norm=bbox_norm, score=score, mask_iou=mask_iou))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


def segment_images(
    processor: Any,
    images: Sequence[Image.Image],
    prompt: str,
    max_candidates: int,
) -> list[list[Candidate]]:
    """Run promptable segmentation over N images. Blocking — call in a thread.

    SAM 3's pipeline (``set_image`` → text prompt → grounding head) binds
    per-image state, so images are processed sequentially; the win from
    passing several at once is at the orchestration layer (one lock
    acquire, one thread hop, one HTTP round trip for N images), not in
    the forward itself.

    Args:
        processor: A ``Sam3Processor`` acquired from :class:`ProcessorPool`.
        images: RGB images to segment.
        prompt: The text prompt, per request. Nothing here supplies a
            default — the concept being segmented is a deployment
            decision, not a property of this server.
        max_candidates: Top-K per image, by score descending.

    Returns:
        One candidate list per input image, aligned by index.
    """
    import torch as _torch

    out: list[list[Candidate]] = []
    with _torch.inference_mode(), _autocast_ctx():
        for image in images:
            width, height = image.size
            state = processor.set_image(image)
            output = _forward_with_cached_prompt(processor, state, prompt)
            out.append(_candidates_from_output(output, width, height, max_candidates))
    return out


__all__ = [
    'Candidate',
    'ProcessorPool',
    'device_name',
    'load_one_instance',
    'load_processors',
    'segment_images',
]
