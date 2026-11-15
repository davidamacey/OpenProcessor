"""The frame engine as ingest's decoder (#217): decode once on the GPU, derive every input there.

Every photo of a curation ingest goes to the frame engine (``frame_engine/``, the ``frame-engine``
service) with one plan: the engine decodes it once (nvJPEG for a baseline JPEG, a CPU decode
inside the engine for every other image), runs the detector on the letterbox, embeds the
whole-frame square and every detection's square with PE, computes the blur sums and encodes the
crop JPEGs; only boxes, vectors, sums and JPEGs come back (:mod:`src.clients.frame_engine`, one
request per image with a plan naming the profile's detector and the PE encoder). The API turns the
detector rows into items with the same :meth:`WholeImageDetector.decode_primary_row` the host
stage uses and reads every other per-image product from the returned :class:`GpuDecodeFrame`. It
never decodes the photo.

There is no fallback to API-side decoding: a photo the engine cannot answer ends as a value,
:class:`EngineFailure`, and the ingest maps it to an error kind (:func:`engine_failure_kind`):
``engine_unavailable`` for an engine that is down, not ready or still overloaded after its
deadline, ``decode_failed`` for a file the engine cannot read, ``detector_infer`` for a branch
that errored. A deployment with ``OP_INGEST_GPU_DECODE=0``, a project with a secondary detector
and one whose letterbox fill is not one grey run the host stage instead
(:func:`engine_stage_applies`, :mod:`src.services.curation.ingest_stage`); each photo's route
(:mod:`src.domain.decode_route`) is stored on its images document as ``decode_route``.

The engine's readiness verdict is kept per detector shape per process for ``READY_TTL_S`` seconds
and dropped on the first call that fails after its retries, so a warm worker sees an engine that
died later and one that came back.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from src.clients import frame_engine
from src.clients.pe_encoder import PE_IMAGE_MODEL, _l2_normalize
from src.config.ingest_profiles import model_class_name
from src.core.logging import get_logger
from src.domain import frame_engine_wire as wire
from src.services.curation.detector_vocabulary import detector_labels
from src.services.curation.ingest_models import (
    ERROR_KIND_DECODE_FAILED,
    ERROR_KIND_DETECTOR_INFER,
    ERROR_KIND_ENGINE_UNAVAILABLE,
)
from src.services.curation.ops_metrics import count_decode_route, count_engine_probe
from src.services.detection.crop_quality import variance_from_sums
from src.utils.class_names import normalize_class_name
from src.utils.pe_preprocess import PE_SIZE
from src.utils.retry import TRITON_CALL_ERRORS


if TYPE_CHECKING:
    from src.services.curation.ingest import CurationIngestService
    from src.services.curation.item_doc import DetectedItem


logger = get_logger(__name__)

#: Squares per encoder call inside the engine (the PE engine's max batch, as the prep model used).
PE_BATCH = 32
#: The answer when the engine stayed overloaded through the client's deadline: the status the
#: engine gives when its device-memory wait ran out.
OVERLOADED = 'gpu_memory'
#: The call itself failed (engine down, a transport error).
CALL_FAILED = 'call_failed'
#: The engine answered ``ok`` for the photo but a requested branch errored.
BRANCH_FAILED = 'branch_failed'
#: How long a positive readiness verdict is trusted.
READY_TTL_S = 5.0

#: Engine statuses that are the engine's load, not the file: a retry later can succeed.
_TRANSIENT_REASONS = frozenset({OVERLOADED, CALL_FAILED})


@dataclass
class GpuDecodeFrame:
    """One photo's products from the engine; per-detection rows are indexed by the item objects
    :func:`run_engine_stage` built (``det_of``)."""

    width: int
    height: int
    whole_embedding: np.ndarray
    crop_embeddings: np.ndarray | None
    blur_sums: np.ndarray
    crop_jpegs: list[bytes]
    det_of: dict[int, int] = field(default_factory=dict)
    embedded: np.ndarray | None = None
    """Per detection, whether the engine embedded its crop (a plan with ``embed_classes`` skips
    the other classes); ``crop_embeddings`` has a row per detection either way."""

    def _row(self, item: DetectedItem) -> int:
        return self.det_of[id(item)]

    def full_var(self) -> float:
        n, total, squares = (int(v) for v in self.blur_sums[0])
        return variance_from_sums(n, total, squares)

    def box_var(self, item: DetectedItem) -> float | None:
        n, total, squares = (int(v) for v in self.blur_sums[1 + self._row(item)])
        return variance_from_sums(n, total, squares) if n > 0 else None

    def crop_embedding(self, item: DetectedItem) -> np.ndarray | None:
        if self.crop_embeddings is None:
            return None
        row = self._row(item)
        if self.embedded is not None and not self.embedded[row]:
            return None
        typed: np.ndarray | None = self.crop_embeddings[row]
        return typed

    def crop_jpeg(self, item: DetectedItem) -> bytes:
        return self.crop_jpegs[self._row(item)]


@dataclass(frozen=True)
class EngineStaged:
    """A photo the engine answered: its products, its items, and the route its pixels took."""

    frame: GpuDecodeFrame
    items: list[DetectedItem]
    decode_route: str


@dataclass(frozen=True)
class EngineFailure:
    """The engine gave no usable answer for one photo: ``reason`` is :data:`OVERLOADED`,
    :data:`CALL_FAILED`, :data:`BRANCH_FAILED` or a status of
    :data:`src.domain.decode_route.PREP_STATUSES`; ``detail`` says it in words."""

    reason: str
    detail: str


def engine_failure_kind(failure: EngineFailure) -> str:
    """The ingest error kind of an engine failure: transient when a retry later can succeed."""
    if failure.reason in _TRANSIENT_REASONS:
        return ERROR_KIND_ENGINE_UNAVAILABLE
    if failure.reason == BRANCH_FAILED:
        return ERROR_KIND_DETECTOR_INFER
    return ERROR_KIND_DECODE_FAILED


#: What a readiness probe can raise besides the Triton client errors: a pool not initialized.
_PROBE_ERRORS: tuple[type[Exception], ...] = (*TRITON_CALL_ERRORS, RuntimeError)
_READY: dict[tuple[str, int, int], float] = {}


def embed_spec() -> wire.EmbedSpec:
    return wire.EmbedSpec(model=PE_IMAGE_MODEL, size=PE_SIZE, batch=PE_BATCH)


def detector_spec(service: CurationIngestService) -> wire.DetectorSpec | None:
    """The profile's detector as the engine runs it; ``None`` when the letterbox fill is not one
    grey (the engine fills the canvas with one value)."""
    p = service.profile
    if len(set(p.letterbox_fill)) != 1:
        return None
    return wire.DetectorSpec(p.detector_model, p.input_size, p.letterbox_fill[0])


def engine_stage_applies(service: CurationIngestService) -> bool:
    """Whether this service's photos take the engine stage: the deployment enables it, the service
    has an engine client, no secondary detector (that one needs the host frame) and a one-grey
    letterbox fill. Otherwise they take the host stage."""
    if (
        not service.config.ingest_gpu_decode
        or service.frame_engine is None
        or service.detector.secondary_profile is not None
    ):
        return False
    if detector_spec(service) is None:
        logger.warning('ingest_engine_fill_not_uniform', fill=service.profile.letterbox_fill)
        return False
    return True


async def engine_ready(service: CurationIngestService) -> bool:
    """Whether the engine can take photos now: it is ready and the detector and PE encoder are
    loaded in Triton (explicit model control: they are loaded here when they are not). A ready
    verdict is kept per detector shape per process for :data:`READY_TTL_S` seconds; a failed probe
    is not kept, and a failed engine call drops the verdicts (:func:`call_engine`)."""
    det = detector_spec(service)
    if det is None or service.frame_engine is None:
        return False
    key = (det.model, det.input_size, det.letterbox_fill)
    verdict = _READY.get(key)
    if verdict is not None and time.monotonic() - verdict < READY_TTL_S:
        return True
    readiness = await frame_engine.probe_ready(service.frame_engine)
    count_engine_probe(readiness.outcome)
    if readiness.outcome == frame_engine.PROBE_NOT_READY:
        logger.warning('ingest_engine_not_ready')
    elif readiness.outcome == frame_engine.PROBE_UNDELIVERED:
        logger.warning(
            'ingest_engine_probe_undelivered',
            attempts=readiness.attempts,
            code=readiness.code,
            error=readiness.error,
        )
    elif readiness.outcome == frame_engine.PROBE_RECOVERED:
        logger.info('ingest_engine_probe_recovered', attempts=readiness.attempts)
    if not readiness.ready:
        _READY.pop(key, None)
        return False
    try:
        for model in (det.model, PE_IMAGE_MODEL):
            if not await service.triton_pool.is_model_ready(model):
                await service.triton_pool.load_model(model)
    except _PROBE_ERRORS as exc:
        logger.warning('ingest_engine_unavailable', error=str(exc))
        _READY.pop(key, None)
        return False
    _READY[key] = time.monotonic()
    return True


async def call_engine(
    pool: Any, data: bytes, plan: wire.EnginePlan, boxes: np.ndarray | None = None, *, caller: str
) -> frame_engine.EngineAnswer | EngineFailure:
    """One frame-engine request for the encoded image ``data``: the answer when the image was
    decoded and every branch ran, else why not."""
    try:
        answer = await frame_engine.process(pool, data, plan, boxes)
    except frame_engine.EngineCallError as exc:
        _READY.clear()  # the next batch asks the engine again
        logger.warning(
            'frame_engine_call_failed', caller=caller, overloaded=exc.overloaded, error=str(exc)
        )
        return EngineFailure(OVERLOADED if exc.overloaded else CALL_FAILED, str(exc))
    if answer.status != wire.STATUS_OK:
        return EngineFailure(answer.status, f'the engine could not read the image: {answer.status}')
    if not answer.ok:
        logger.warning('frame_engine_branch_failed', caller=caller, branches=answer.branches)
        return EngineFailure(BRANCH_FAILED, f'an engine branch failed: {answer.branches}')
    return answer


async def call_frame_engine(
    pool: Any, data: bytes, plan: wire.EnginePlan, boxes: np.ndarray | None = None, *, caller: str
) -> frame_engine.EngineAnswer | str:
    """:func:`call_engine` for the callers that fall back to the host on a failure (the embed
    reprocess pass): the answer, else the reason as a string."""
    out = await call_engine(pool, data, plan, boxes, caller=caller)
    if isinstance(out, EngineFailure):
        return CALL_FAILED if out.reason == BRANCH_FAILED else out.reason
    return out


def _embed_class_ids(service: CurationIngestService) -> frozenset[int] | None:
    """The detector class ids whose crops can be embedded at all: the labels that pass the detect
    filter's class lists and, for the policy ``selected``, its class list (class identity by NAME;
    the ids are only the engine's addressing). ``None`` when no list restricts the classes, or the
    detector has no label names to match them with."""
    policy = service.policy
    keep: set[str] | None = None
    for names in (
        policy.detect.classes,
        policy.embedding.classes if policy.embedding.mode == 'selected' else None,
    ):
        if names:
            wanted = {normalize_class_name(n) for n in names}
            keep = wanted if keep is None else keep & wanted
    drop = {normalize_class_name(n) for n in policy.detect.exclude_classes}
    if keep is None and not drop:
        return None
    labels = detector_labels(service.profile)
    if not labels:
        return None
    ids = set()
    for class_id in range(max(label.class_id for label in labels) + 1):
        slug = normalize_class_name(model_class_name(service.profile, class_id) or str(class_id))
        if (keep is None or slug in keep) and slug not in drop:
            ids.add(class_id)
    return frozenset(ids)


def ingest_engine_plan(service: CurationIngestService) -> wire.EnginePlan:
    """The one plan every photo of an ingest carries: detect, the whole-frame vector, and the crop
    vectors unless the policy defers them (``lazy``, ``none``) or no class can be embedded."""
    det = detector_spec(service)
    if det is None:
        raise RuntimeError('the engine plan needs a one-grey letterbox fill (engine_stage_applies)')
    branches: list[wire.EngineBranch] = ['detect', 'embed_frame']
    embed_classes = None
    if service.policy.embedding.mode not in ('lazy', 'none'):
        branches.append('embed_crops')
        embed_classes = _embed_class_ids(service)
    return wire.EnginePlan(
        branches=tuple(branches),
        detector=det,
        embed=embed_spec(),
        embed_classes=embed_classes,
        crops='inline',
    )


def _crop_rows(out: dict[str, Any], n: int, dim: int) -> tuple[np.ndarray, np.ndarray | None]:
    """The normalized crop vectors with one row per detection, and which rows the engine embedded.
    The engine answers only the picked rows (``crop_embedded`` marks them); the others are zero
    rows no reader may use."""
    picked = _l2_normalize(out[wire.CROP_EMBEDDINGS].reshape(-1, dim).astype(np.float32))
    mask = out.get(wire.CROP_EMBEDDED)
    if mask is None:
        return picked, None
    flags = np.asarray(mask, np.bool_).reshape(-1)
    rows = np.zeros((n, dim), np.float32)
    rows[flags] = picked
    return rows, flags


async def run_engine_stage(
    service: CurationIngestService, data: bytes
) -> EngineStaged | EngineFailure:
    """The engine's products and the detected items for one image, or why it has none."""
    answer = await call_engine(
        service.frame_engine, data, ingest_engine_plan(service), caller='ingest'
    )
    if isinstance(answer, EngineFailure):
        return answer
    plan_has_crops = wire.CROP_EMBEDDINGS in answer.outputs
    out = answer.outputs
    width, height = (int(v) for v in out[wire.SIZE])
    scale, pad_w, pad_h = (float(v) for v in out[wire.LETTERBOX])
    boxes = out[wire.DET_BOXES].reshape(-1, 4)
    n = boxes.shape[0]
    items = service.detector.decode_primary_row(
        boxes, out[wire.DET_SCORES], out[wire.DET_CLASSES], scale, (pad_w, pad_h)
    )
    whole = out[wire.WHOLE_EMBEDDING].astype(np.float32)
    crops, embedded = _crop_rows(out, n, whole.shape[0]) if plan_has_crops and n else (None, None)
    frame = GpuDecodeFrame(
        width=width,
        height=height,
        whole_embedding=_l2_normalize(whole),
        crop_embeddings=crops,
        blur_sums=out[wire.BLUR_SUMS].reshape(-1, 3),
        crop_jpegs=list(out[wire.CROP_JPEGS]),
        det_of={id(item): i for i, item in enumerate(items)},
        embedded=embedded,
    )
    return EngineStaged(frame, items, answer.decode_route)


def count_route(route: str) -> None:
    count_decode_route(route, 'ingest')
