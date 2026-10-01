"""``POST /prompt_packs/test`` and ``POST /region_profiles/test`` (W5,
any_domain_plan.md §5): run a draft or saved pack / profile (and VLM
endpoint) against stored crops of the bound project and return the preview.

Read-only: neither route writes to any index, bumps a revision or touches the
crop cache beyond reading. Both go through the one VLM gate (mode ``test``:
external-images acknowledgement, SSRF policy, secrets by reference), are
size-capped (one upstream request's worth of crops), concurrency-capped per
process (429 ``test_busy``) and wall-clock bounded (504 ``test_timeout``).
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from src.clients.curation_opensearch import mget_crops
from src.routers.curation._common import OpenSearchDep, get_class_registry, items_index, router
from src.routers.curation._config_common_models import ValidationReport, api_error
from src.routers.curation._config_test_models import (
    PackTestCropResult,
    PackTestPackRef,
    PackTestPrompt,
    PackTestRequest,
    PackTestResponse,
    PackTestVlmRef,
    RegionTestCandidate,
    RegionTestLeg,
    RegionTestProfileRef,
    RegionTestRequest,
    RegionTestResponse,
    RegionTestVerify,
)
from src.routers.curation.pipeline_vlm import labeler_unavailable, resolve_test_vlm
from src.services.config_store import (
    get_config_store,
    packs as pack_store,
    profiles as profile_store,
)
from src.services.config_store.pack_validation import validate_pack
from src.services.config_store.profile_validation import validate_profile
from src.services.curation.config_test_limits import (
    ConfigTestBusyError,
    ConfigTestTimeoutError,
    bounded,
    reserve_slot,
)
from src.services.curation.crop_bytes import load_item_crop_jpeg, resolved_image_path
from src.services.curation.pack_test_run import (
    CropImageUnavailableError,
    CropInput,
    NoRegionBoxError,
    TooManyCropsError,
    run_pack_test,
)
from src.services.curation.region_class import item_classes
from src.services.curation.region_test_run import Leg, VerifyContext, run_region_test
from src.services.curation.wire import item_source_excludes
from src.services.detection.segmenter_http import first_segmenter_url
from src.services.labeling.vlm_endpoints import VlmEndpointUnavailableError
from src.services.labeling.vlm_factory import labeler_for
from src.services.labeling.vlm_labeler import VlmTransportError
from src.services.labeling.vlm_prompts import PromptPack, active_prompt_pack, prompt_pack_stamp


if TYPE_CHECKING:
    from src.config import DetectionProfile
    from src.routers.curation._prompt_pack_models import PromptPackBody
    from src.services.labeling.vlm_endpoints import VlmEndpoint
    from src.services.labeling.vlm_labeler import VlmLabeler


# ---- shared resolution ---------------------------------------------------------


def _registry() -> tuple[list[str], dict[str, int]]:
    """The labelable class names and ``name -> class_id`` of the bound project."""
    labelable = item_classes(get_class_registry().load().classes)
    return [c.class_name for c in labelable], {c.class_name: int(c.class_id) for c in labelable}


def _active_profile_stamp() -> tuple[str | None, int | None]:
    ref = get_config_store().current.active_profile
    return (ref[0], ref[1]) if isinstance(ref, tuple) else (None, None)


async def _resolve_pack(
    opensearch: Any,
    *,
    name: str | None,
    revision: int | None,
    draft: PromptPackBody | None,
    profile: DetectionProfile | None,
) -> tuple[PromptPack, int | None, bool]:
    """``(pack, resolved revision, is_draft)``. A draft is validated first
    (422 ``pack_invalid`` + report); an unknown name is 422 ``unknown_pack``,
    a name with no such revision 422 ``unknown_revision``."""
    if name is not None and draft is not None:
        raise api_error(422, 'validation_failed', 'give a saved pack or a draft, not both')
    if draft is not None:
        data = {**draft.model_dump(), 'name': 'draft'}
        report = validate_pack(None, data, profile=profile, class_names=frozenset(_registry()[0]))
        if not report.ok:
            raise api_error(422, 'pack_invalid', 'the draft pack has errors', report=report)
        return PromptPack.from_dict(data), None, True
    if name is None:
        pack = active_prompt_pack()
        return pack, None, False
    record = pack_store.build_record(name, revision=revision)
    if record is None and revision is not None:
        record = await pack_store.get_revision_record(opensearch, name, revision)
    if record is not None and revision is not None and record.revision != revision:
        record = None  # a built-in or file pack has no numbered revisions
    if record is None:
        known = pack_store.build_record(name) is not None
        message = f'{name!r}' + (f'@{revision}' if revision else '') + ' is unknown'
        raise api_error(422, 'unknown_revision' if known else 'unknown_pack', message)
    return PromptPack.from_dict({**record.body, 'name': name}), record.revision, False


def _vlm_ref(endpoint: VlmEndpoint, *, draft: bool, labeler: VlmLabeler) -> PackTestVlmRef:
    return PackTestVlmRef(
        endpoint=labeler.identity.endpoint_ref,
        model=labeler.identity.model,
        name=None if draft else endpoint.name,
        revision=None if draft else endpoint.revision,
        draft=draft,
    )


def _labeler(endpoint: VlmEndpoint, pack: PromptPack) -> VlmLabeler:
    try:
        return labeler_for(endpoint, pack)
    except VlmEndpointUnavailableError as exc:
        raise labeler_unavailable(exc) from exc


def _guard_errors(exc: Exception) -> Any:
    """The ``api_error`` for a run-time failure shared by both routes."""
    if isinstance(exc, ConfigTestBusyError):
        return api_error(429, 'test_busy', 'too many test runs are in progress; retry shortly')
    if isinstance(exc, ConfigTestTimeoutError):
        return api_error(504, 'test_timeout', 'the test run took too long and was stopped')
    if isinstance(exc, VlmTransportError):
        return api_error(502, 'vlm_transport_error', f'the VLM endpoint failed: {exc}')
    if isinstance(exc, CropImageUnavailableError):
        return api_error(404, 'image_not_found', f"crop {exc.crop_id!r}'s image could not be read")
    return exc


async def _load_crops(opensearch: Any, crop_ids: list[str]) -> list[CropInput]:
    """The bound project's crops, in request order. 404 ``crop_not_found``
    naming every missing id; a crop whose stored image path is not servable
    is 404 ``image_not_found``."""
    ids = list(dict.fromkeys(crop_ids))
    docs = await mget_crops(
        opensearch, ids, index=items_index(), source_excludes=item_source_excludes()
    )
    missing = [i for i in ids if i not in docs]
    if missing:
        raise api_error(
            404, 'crop_not_found', f'no such crop: {", ".join(missing)}', crop_ids=missing
        )
    crops: list[CropInput] = []
    for crop_id in ids:
        source = docs[crop_id].get('_source') or {}
        bbox = source.get('bbox_norm')
        image_path = source.get('image_path') or ''
        if not image_path or not bbox or len(bbox) != 4:
            raise api_error(404, 'image_not_found', f'crop {crop_id!r} has no readable image')
        try:
            path = resolved_image_path(image_path)
        except HTTPException as exc:
            raise api_error(404, 'image_not_found', f'crop {crop_id!r}: {exc.detail}') from exc
        crops.append(CropInput(crop_id, source, path, (bbox[0], bbox[1], bbox[2], bbox[3])))
    return crops


def _pack_ref(pack: PromptPack, revision: int | None, draft: bool) -> PackTestPackRef:
    return PackTestPackRef(name=None if draft else pack.name, revision=revision, draft=draft)


def _profile_for(name: str | None) -> DetectionProfile | None:
    from src.services.detection.profile_registry import get_active_region_profile, get_profiles

    return get_active_region_profile() if name is None else get_profiles().get(name)


# ---- POST /prompt_packs/test -----------------------------------------------------


@router.post('/prompt_packs/test', response_model=PackTestResponse)
async def test_prompt_pack(body: PackTestRequest, opensearch: OpenSearchDep) -> PackTestResponse:
    """Run one call of a pack over stored crops and preview each item's write.

    ``parse_ok: false`` (the reply did not parse) is a 200 result, not an
    error. 422 ``unknown_pack`` / ``unknown_revision`` / ``pack_invalid``
    (+ report) / ``unknown_vlm`` / ``vlm_external_not_acknowledged`` /
    ``too_many_crops`` / ``no_box_to_verify``; 404 ``crop_not_found``;
    409 ``vlm_not_configured``; 429 ``test_busy``; 502 ``vlm_transport_error``.
    Writes nothing."""
    if not body.crop_ids:
        raise api_error(422, 'validation_failed', 'crop_ids must name at least one crop')
    await get_config_store().ensure_fresh(opensearch)
    profile = _profile_for(body.profile_name)
    if body.profile_name is not None and profile is None:
        raise api_error(422, 'unknown_profile', f'{body.profile_name!r} is not a known profile')
    pack, revision, is_draft = await _resolve_pack(
        opensearch,
        name=body.pack_name,
        revision=body.pack_revision,
        draft=body.draft,
        profile=profile,
    )
    endpoint = await resolve_test_vlm(
        opensearch,
        name=body.vlm_name,
        revision=body.vlm_revision,
        draft=body.vlm_draft,
        acknowledge_external=body.acknowledge_external,
        pack=pack,
        profile=profile,
    )
    labeler = _labeler(endpoint, pack)
    capacity = (
        labeler.open_images_per_call
        if body.call == 'open_classify'
        else labeler.max_images_per_call
    )
    if body.call == 'combined' and profile is None:
        raise api_error(
            409, 'no_active_profile', 'combined needs a region profile (active or profile_name)'
        )
    class_names, name_to_id = _registry()
    if body.class_names is not None:
        class_names = body.class_names
    crops = await _load_crops(opensearch, body.crop_ids)
    stamp = _active_profile_stamp() if body.profile_name is None else (body.profile_name, None)
    try:
        with reserve_slot('vlm'):
            run = await bounded(
                run_pack_test(
                    labeler,
                    body.call,
                    crops,
                    use_region_box=body.use_region_box,
                    class_names=class_names,
                    name_to_id=name_to_id,
                    profile=profile,
                    pack_stamp=prompt_pack_stamp(pack, revision=revision),
                    stamp_profile=stamp,
                    capacity=capacity,
                )
            )
    except NoRegionBoxError as exc:
        raise api_error(
            422, 'no_box_to_verify', 'region_verify needs a stored box open to a machine verdict'
        ) from exc
    except TooManyCropsError as exc:
        raise api_error(
            422,
            'too_many_crops',
            f'region_verify would send {exc.n} images; at most {capacity}',
            limit=capacity,
        ) from exc
    except (
        ConfigTestBusyError,
        ConfigTestTimeoutError,
        VlmTransportError,
        CropImageUnavailableError,
    ) as exc:
        raise _guard_errors(exc) from exc
    result = run.probe
    return PackTestResponse(
        call=body.call,
        pack=_pack_ref(pack, revision, is_draft),
        vlm=_vlm_ref(endpoint, draft=body.vlm_draft is not None, labeler=labeler),
        prompt=PackTestPrompt(system=result.prompt_system, user_text=result.prompt_user_text),
        raw_reply=result.raw_reply,
        reasoning=result.reasoning,
        latency_ms=result.latency_ms,
        parse_ok=result.parse_ok,
        parse_error=result.parse_error,
        results=[
            PackTestCropResult(
                crop_id=r.crop_id,
                box_id=r.box_id,
                parsed=r.parsed,
                preview_item=r.preview,
                skipped=r.skipped,
            )
            for r in run.results
        ],
        validation=validate_pack(
            None, pack.to_dict(), profile=profile, class_names=frozenset(_registry()[0])
        ),
    )


# ---- POST /region_profiles/test --------------------------------------------------


async def _resolve_profile(
    opensearch: Any, body: RegionTestRequest
) -> tuple[DetectionProfile, int | None, str | None, ValidationReport]:
    """``(profile, revision, stamped name, validation)`` -- the stamp name is
    ``None`` for a draft (never saved). The segmenter-prompt override is
    applied to the body BEFORE validation, so it is validated too."""
    from src.services.detection.profile_registry import (
        get_active_region_profile,
        region_profile_from_dict,
    )

    if body.profile_name is not None and body.draft is not None:
        raise api_error(422, 'validation_failed', 'give a saved profile or a draft, not both')
    if body.draft is None and body.profile_name is None:
        active = get_active_region_profile()
        if active is None:
            raise api_error(
                409, 'no_active_profile', 'no region profile is active; name one or send a draft'
            )
        revision = _active_profile_stamp()[1]
        if body.segmenter_text_prompt is not None:
            active = dataclasses.replace(active, segmenter_text_prompt=body.segmenter_text_prompt)
        return active, revision, active.name, ValidationReport(ok=True)
    if body.draft is not None:
        data: dict[str, Any] = body.draft.model_dump()
        name, revision, stamp = 'draft', None, None
    else:
        assert body.profile_name is not None
        record = profile_store.build_record(body.profile_name, revision=body.profile_revision)
        if record is None and body.profile_revision is not None:
            record = await profile_store.get_revision_record(
                opensearch, body.profile_name, body.profile_revision
            )
        if record is None:
            raise api_error(422, 'unknown_profile', f'{body.profile_name!r} is not a known profile')
        data, name, revision, stamp = (
            dict(record.body),
            body.profile_name,
            record.revision,
            body.profile_name,
        )
    if body.segmenter_text_prompt is not None:
        data['segmenter_text_prompt'] = body.segmenter_text_prompt
    report = await validate_profile(
        name,
        data,
        segmenter_health=_segmenter_health,
        class_names=frozenset(_registry()[0]),
        project_slug=_project_slug(),
    )
    if not report.ok:
        raise api_error(422, 'profile_invalid', 'the profile has errors', report=report)
    return region_profile_from_dict({**data, 'name': name}, source=name), revision, stamp, report


async def _segmenter_health() -> tuple[str, str | None]:
    from src.routers.curation._models_segmenter import _segmenter_health as health

    url = first_segmenter_url()
    if url is None:
        return 'unavailable', 'OP_SEGMENTER_URL is not configured'
    return await health(url)


def _project_slug() -> str | None:
    from src.config import get_curation_config

    return get_curation_config().project_slug


def _leg_wire(leg: Leg) -> RegionTestLeg:
    return RegionTestLeg(
        leg=leg.leg,  # type: ignore[arg-type]
        status=leg.status,  # type: ignore[arg-type]
        reason=leg.reason,
        elapsed_ms=leg.elapsed_ms,
        candidates=[RegionTestCandidate(**c) for c in leg.candidates],
    )


@router.post('/region_profiles/test', response_model=RegionTestResponse)
async def test_region_profile(
    body: RegionTestRequest, opensearch: OpenSearchDep
) -> RegionTestResponse:
    """Run a profile's detector and segmenter legs over one stored crop, show
    every raw candidate with what the worker's selection does with it, and the
    item as the worker would leave it (optionally with the combined VLM call).

    A leg that fails is reported on that leg (200); only when no leg ran at all
    is it 502 ``detector_error`` / ``segmenter_error``. 404 ``crop_not_found``;
    409 ``no_active_profile``; 422 ``profile_invalid`` (+ report); 429
    ``test_busy``. Writes nothing."""
    await get_config_store().ensure_fresh(opensearch)
    profile, revision, stamp_name, report = await _resolve_profile(opensearch, body)
    verify: VerifyContext | None = None
    verify_ref: tuple[PackTestPackRef, PackTestVlmRef] | None = None
    if body.verify:
        pack, pack_revision, pack_draft = await _resolve_pack(
            opensearch,
            name=body.prompt_pack_name,
            revision=body.prompt_pack_revision,
            draft=body.prompt_pack_draft,
            profile=profile,
        )
        endpoint = await resolve_test_vlm(
            opensearch,
            name=body.vlm_name,
            revision=body.vlm_revision,
            draft=body.vlm_draft,
            acknowledge_external=body.acknowledge_external,
            pack=pack,
            profile=profile,
        )
        labeler = _labeler(endpoint, pack)
        class_names, name_to_id = _registry()
        verify = VerifyContext(
            labeler=labeler,
            pack_stamp=prompt_pack_stamp(pack, revision=pack_revision),
            class_names=class_names,
            name_to_id=name_to_id,
        )
        verify_ref = (
            _pack_ref(pack, pack_revision, pack_draft),
            _vlm_ref(endpoint, draft=body.vlm_draft is not None, labeler=labeler),
        )
    (crop,) = await _load_crops(opensearch, [body.crop_id])
    jpeg = load_item_crop_jpeg(crop.crop_id, str(crop.image_path), crop.bbox)
    if jpeg is None:
        raise api_error(404, 'image_not_found', f"crop {crop.crop_id!r}'s image could not be read")
    from src.main import get_async_triton_pool

    prompt = (
        body.segmenter_text_prompt
        if body.segmenter_text_prompt is not None
        else profile.segmenter_text_prompt
    )
    try:
        with contextlib.ExitStack() as slots:
            slots.enter_context(reserve_slot('segmenter'))
            if verify is not None:
                slots.enter_context(reserve_slot('vlm'))
            run = await bounded(
                run_region_test(
                    source=crop.source,
                    crop_id=crop.crop_id,
                    jpeg=jpeg,
                    profile=profile,
                    stamp_name=stamp_name,
                    stamp_revision=revision,
                    segmenter_prompt=prompt,
                    segmenter_url=first_segmenter_url(),
                    triton_pool=get_async_triton_pool,
                    verify=verify,
                )
            )
    except (ConfigTestBusyError, ConfigTestTimeoutError, VlmTransportError) as exc:
        raise _guard_errors(exc) from exc
    ran = [leg for leg in run.legs if leg.status != 'skipped']
    if ran and all(leg.status == 'error' for leg in ran):
        failed = ran[0]
        raise api_error(
            502,
            'detector_error' if failed.leg == 'detector' else 'segmenter_error',
            f'the {failed.leg} leg failed: {failed.reason}',
        )
    verify_block = None
    if run.probe is not None and verify_ref is not None:
        verify_block = RegionTestVerify(
            pack=verify_ref[0],
            vlm=verify_ref[1],
            prompt=PackTestPrompt(
                system=run.probe.prompt_system, user_text=run.probe.prompt_user_text
            ),
            raw_reply=run.probe.raw_reply,
            reasoning=run.probe.reasoning,
            latency_ms=run.probe.latency_ms,
            parse_ok=run.probe.parse_ok,
            parse_error=run.probe.parse_error,
        )
    parents = profile.parent_classes
    return RegionTestResponse(
        crop_id=crop.crop_id,
        profile=RegionTestProfileRef(
            name=stamp_name, revision=revision, draft=body.draft is not None
        ),
        item_eligible=not parents or str(crop.source.get('class_name') or '') in parents,
        legs=[_leg_wire(leg) for leg in run.legs],
        verify=verify_block,
        preview_basis=run.basis,  # type: ignore[arg-type]
        preview_item=run.preview,  # type: ignore[arg-type]
        validation=report,
    )
