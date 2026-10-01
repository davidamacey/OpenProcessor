"""Run one pack call over stored crops and preview each item's write (W5,
``POST /prompt_packs/test``).

The VLM call is the production one, observed
(:func:`~src.services.labeling.vlm_probe.probe`); each parsed answer is turned
into the item write by the same functions the production writer calls
(``resolve_combined_reply`` -> ``box_pass_update`` -> ``finalize_region_write``
for the combined call, ``label_batch_update`` / ``label_batch_merge`` for the
class calls, ``verify_regions_update`` for region verify), then laid over the
stored item. Nothing is written.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from src.config import get_curation_config, get_region_fields
from src.services.curation.class_write_guard import class_state_token, class_write_locked
from src.services.curation.crop_bytes import (
    load_item_crop_jpeg,
    load_region_jpeg,
    load_vlm_item_jpeg,
)
from src.services.curation.label_batch_write import label_batch_merge, label_batch_update
from src.services.curation.region_box_pass import (
    box_pass_update,
    finalize_region_write,
    worker_stamps,
)
from src.services.curation.region_boxes import read_boxes
from src.services.curation.region_preview import preview_item
from src.services.curation.region_verify import BoxVerdict, verifiable_boxes, verify_regions_update
from src.services.labeling.vlm_labeler import resolve_class_name
from src.services.labeling.vlm_probe import ProbeCrop, probe


if TYPE_CHECKING:
    from pathlib import Path

    from src.config import DetectionProfile
    from src.services.labeling.vlm_labeler import VlmLabeler
    from src.services.labeling.vlm_probe import ProbeCall, ProbeResult


class CropImageUnavailableError(RuntimeError):
    """A crop's source image could not be read (``404 image_not_found``)."""

    def __init__(self, crop_id: str) -> None:
        super().__init__(crop_id)
        self.crop_id = crop_id


class NoRegionBoxError(RuntimeError):
    """``region_verify`` needs at least one stored box open to a machine
    verdict (``422 no_region_box``)."""


class TooManyCropsError(RuntimeError):
    """The call would send more images than one upstream request carries
    (``422 too_many_crops``)."""

    def __init__(self, n: int) -> None:
        super().__init__(n)
        self.n = n


def _check_capacity(n_images: int, capacity: int) -> None:
    if n_images > capacity:
        raise TooManyCropsError(n_images)


@dataclass(frozen=True)
class CropInput:
    crop_id: str
    source: dict[str, Any]
    #: The stored image path, already resolved under the crop root.
    image_path: Path
    bbox: tuple[float, float, float, float]


@dataclass
class CropResult:
    crop_id: str
    box_id: str | None = None
    parsed: dict[str, Any] | bool | None = None
    preview: dict[str, Any] | None = None
    skipped: str | None = None


@dataclass
class PackTestRun:
    probe: ProbeResult
    results: list[CropResult] = field(default_factory=list)


def jsonable(value: Any) -> dict[str, Any] | bool | None:
    """A parsed answer as plain JSON."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode='json')
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return dict(value)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _jpeg(crop: CropInput) -> bytes:
    jpeg = load_item_crop_jpeg(crop.crop_id, str(crop.image_path), crop.bbox)
    if jpeg is None:
        raise CropImageUnavailableError(crop.crop_id)
    return jpeg


async def _combined(
    labeler: VlmLabeler,
    crops: list[CropInput],
    *,
    use_region_box: str,
    class_names: list[str],
    name_to_id: dict[str, int],
    profile: DetectionProfile,
    pack_stamp: str,
    stamp_profile: tuple[str | None, int | None],
    capacity: int,
) -> PackTestRun:
    from scripts.curation.worker.combined_resolve import resolve_combined_reply, should_classify
    from scripts.curation.worker.verify import _combined_class_update, task_box_from_stored
    from src.services.detection.region_text_rules import region_text_rules

    _check_capacity(len(crops), capacity)
    F = get_region_fields()
    jpegs = {c.crop_id: _jpeg(c) for c in crops}
    candidates = {
        c.crop_id: (
            [task_box_from_stored(b, item_bbox_norm=c.bbox) for b in read_boxes(c.source, F)]
            if use_region_box == 'current'
            else []
        )
        for c in crops
    }
    result = await probe(
        labeler,
        'combined',
        [
            ProbeCrop(
                crop_id=c.crop_id,
                jpeg=jpegs[c.crop_id],
                region_boxes=[t.bbox_in_crop for t in candidates[c.crop_id]],
            )
            for c in crops
        ],
        class_names=class_names or None,
    )
    run = PackTestRun(probe=result)
    model = labeler.identity.model
    stamps = worker_stamps(
        profile_name=stamp_profile[0],
        profile_revision=stamp_profile[1],
        pack_stamp=pack_stamp,
        vlm_called=True,
        vlm_endpoint=labeler.identity.endpoint_ref,
        vlm_model=model,
    )
    for crop, entry in zip(crops, result.parsed, strict=True):
        reply = entry.value
        item = CropResult(crop.crop_id, parsed=jsonable(reply))
        run.results.append(item)
        if reply is None:
            item.skipped = 'no_usable_answer'
            continue
        names = (
            class_names
            if should_classify(
                class_validated=bool(crop.source.get('class_validated')),
                stored_class_source=str(crop.source.get('class_source') or ''),
                test_holdout=bool(crop.source.get('test_holdout')),
                class_confidence=float(crop.source.get('confidence') or 0.0),
                registry_loaded=bool(class_names),
            )
            else None
        )
        cands = candidates[crop.crop_id]
        if not cands:
            # No box offered: the call only classifies and answers visibility.
            update = _combined_class_update(reply, names, vlm_model=model, name_to_id=name_to_id)
            trace: list[str] = []
        else:
            resolution = await resolve_combined_reply(
                cands,
                reply,
                item_bbox_norm=crop.bbox,
                reverify=True,
                effective_class_names=names,
                name_to_id=name_to_id,
                vlm_model=model,
                profile=profile,
                rules=region_text_rules(profile),
                ocr=None,
                crop_jpeg=jpegs[crop.crop_id],
                crop_id=crop.crop_id,
            )
            if resolution.outcome == 'no_verdict':
                item.skipped = 'no_verdict'
                continue
            assert resolution.status is not None
            box_pass = box_pass_update(
                crop.source,
                resolution.boxes,
                reverify=True,
                merge_machine_boxes=False,
                baseline=read_boxes(crop.source, F),
                status=None,
                empty_status=resolution.status,
            )
            update = {**resolution.extra, **box_pass.update}
            trace = [f'{cands[0].detector}:hit', *resolution.trace]
        write = finalize_region_write(
            update,
            crop.source,
            doc_id=crop.crop_id,
            class_token=class_state_token(crop.source),
            trace=trace,
            stamps=stamps,
        )
        item.preview = preview_item(crop.source, write, crop.crop_id)
    return run


async def _classify(
    labeler: VlmLabeler,
    call: ProbeCall,
    crops: list[CropInput],
    *,
    class_names: list[str],
    name_to_id: dict[str, int],
    pack_stamp: str,
    capacity: int,
) -> PackTestRun:
    _check_capacity(len(crops), capacity)
    cache_dir = get_curation_config().crop_cache_dir
    probe_crops = []
    for c in crops:
        jpeg = load_vlm_item_jpeg(c.crop_id, str(c.image_path), c.bbox, cache_dir=cache_dir)
        if jpeg is None:
            raise CropImageUnavailableError(c.crop_id)
        probe_crops.append(ProbeCrop(crop_id=c.crop_id, jpeg=jpeg))
    result = await probe(labeler, call, probe_crops, class_names=class_names)
    run = PackTestRun(probe=result)
    now = _now()

    def resolve(raw: str, *, confidence: str | None = None) -> str | None:
        return resolve_class_name(raw, name_to_id, confidence=confidence)  # type: ignore[arg-type]

    for crop, entry in zip(crops, result.parsed, strict=True):
        item = CropResult(crop.crop_id, parsed=jsonable(entry.value))
        run.results.append(item)
        if class_write_locked(crop.source):
            item.skipped = 'class_locked'
            continue
        update, _proposal = label_batch_update(
            entry.value,
            name_to_id=name_to_id,
            resolve=resolve,
            now=now,
            pack_stamp=pack_stamp,
            vlm_endpoint=labeler.identity.endpoint_ref,
            vlm_model=labeler.identity.model,
        )
        if update is None:
            item.skipped = 'request_failed'
            continue
        item.preview = preview_item(
            crop.source, label_batch_merge(update, crop.source), crop.crop_id
        )
    return run


async def _region_verify(
    labeler: VlmLabeler, crops: list[CropInput], *, pack_stamp: str, capacity: int
) -> PackTestRun:
    F = get_region_fields()
    probe_crops: list[ProbeCrop] = []
    owner: list[tuple[CropInput, Any]] = []
    for c in crops:
        for box in verifiable_boxes(c.source, F):
            jpeg = load_region_jpeg(str(c.image_path), box.bbox_norm)
            probe_crops.append(ProbeCrop(crop_id=f'{c.crop_id}#{box.box_id}', jpeg=jpeg))
            owner.append((c, box))
    if not probe_crops:
        raise NoRegionBoxError
    _check_capacity(len(probe_crops), capacity)
    result = await probe(labeler, 'region_verify', probe_crops)
    run = PackTestRun(probe=result)
    verdicts: dict[str, list[BoxVerdict]] = {}
    for (crop, box), entry in zip(owner, result.parsed, strict=True):
        run.results.append(
            CropResult(crop.crop_id, box_id=box.box_id, parsed=jsonable(entry.value))
        )
        if entry.value is not None:
            verdicts.setdefault(crop.crop_id, []).append(
                BoxVerdict(
                    box_id=box.box_id,
                    bbox_norm=box.bbox_norm,
                    is_region=bool(entry.value.is_region),
                    confidence=entry.value.confidence,
                    reason=entry.value.reason,
                )
            )
    previews: dict[str, dict[str, Any] | None] = {}
    for crop in crops:
        write = verify_regions_update(
            crop.source,
            verdicts.get(crop.crop_id, []),
            now=_now(),
            pack_stamp=pack_stamp,
            vlm_stamp={
                'vlm_endpoint': labeler.identity.endpoint_ref,
                'vlm_model': labeler.identity.model,
            },
        )
        previews[crop.crop_id] = preview_item(crop.source, write, crop.crop_id) if write else None
    for item in run.results:
        item.preview = previews.get(item.crop_id)
        if item.preview is None:
            item.skipped = 'no_verdict_applies'
    return run


async def _region_visible(
    labeler: VlmLabeler, crops: list[CropInput], *, capacity: int
) -> PackTestRun:
    _check_capacity(len(crops), capacity)
    result = await probe(
        labeler,
        'region_visible',
        [ProbeCrop(crop_id=c.crop_id, jpeg=_jpeg(c)) for c in crops],
    )
    run = PackTestRun(probe=result)
    run.results = [
        CropResult(c.crop_id, parsed=jsonable(e.value))
        for c, e in zip(crops, result.parsed, strict=True)
    ]
    return run


async def run_pack_test(
    labeler: VlmLabeler,
    call: ProbeCall,
    crops: list[CropInput],
    *,
    use_region_box: str,
    class_names: list[str],
    name_to_id: dict[str, int],
    profile: DetectionProfile | None,
    pack_stamp: str,
    stamp_profile: tuple[str | None, int | None],
    capacity: int,
) -> PackTestRun:
    """``capacity`` is the images one upstream request carries (the labeler's
    own cap); more is :class:`TooManyCropsError`."""
    if call == 'combined':
        assert profile is not None, 'combined needs a region profile'
        return await _combined(
            labeler,
            crops,
            use_region_box=use_region_box,
            class_names=class_names,
            name_to_id=name_to_id,
            profile=profile,
            pack_stamp=pack_stamp,
            stamp_profile=stamp_profile,
            capacity=capacity,
        )
    if call in ('classify', 'open_classify'):
        return await _classify(
            labeler,
            call,
            crops,
            class_names=class_names,
            name_to_id=name_to_id,
            pack_stamp=pack_stamp,
            capacity=capacity,
        )
    if call == 'region_verify':
        return await _region_verify(labeler, crops, pack_stamp=pack_stamp, capacity=capacity)
    return await _region_visible(labeler, crops, capacity=capacity)


__all__ = [
    'CropImageUnavailableError',
    'CropInput',
    'CropResult',
    'NoRegionBoxError',
    'PackTestRun',
    'TooManyCropsError',
    'jsonable',
    'run_pack_test',
]
