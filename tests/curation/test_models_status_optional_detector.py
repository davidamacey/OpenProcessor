"""`GET /curation/models/status` must not report a red `not_ready` for a
region detector that simply isn't shipped/installed, when the active
profile also configures a segmenter as its fallback.

Bug: the public `license_plate` example profile names `detector_model:
"license_plate_detector"`, which never ships (no CC/permissively-licensed
plate CNN was found -- see the example profile's own comment). The cascade
still works end to end via the segmenter (SAM 3) + OCR text-hint path (see
`region_dependency_health.stall_reason`'s "a ready segmenter means a down
detector isn't a stall" semantics), but `/models/status` showed a flat red
`not_ready` for a model that was never installed in the first place --
indistinguishable from a real regression (present in Triton's repository
index but failed to load).

Fix: `_build_triton_entry` gains an `optional` flag; `models_status()` sets
it True only for the active profile's region detector when a segmenter is
also configured. When optional and the model is entirely absent from
Triton's `/v2/repository/index`, status is the new `not_installed` value
rather than `not_ready`. Present-but-unloaded/failed stays `not_ready`
unchanged, and every other roster entry's `optional` is always False.
"""

from __future__ import annotations

import httpx
import pytest

from src.services.detection import profile_registry


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_profiles():
    profile_registry._reset_registry_for_tests()
    yield
    profile_registry._reset_registry_for_tests()


def _patch_async_client(monkeypatch: pytest.MonkeyPatch, models_mod, handler) -> None:
    class _FakeAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs.pop('transport', None)
            kwargs['transport'] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(models_mod.httpx, 'AsyncClient', _FakeAsyncClient)


async def test_detector_missing_from_index_with_segmenter_is_optional_not_installed(
    monkeypatch: pytest.MonkeyPatch,
    reference_region_profile: None,
) -> None:
    """license_plate.json: detector_model='license_plate_detector' never
    appears in Triton's repository index; segmenter_name='sam3' is
    configured as the fallback. The detector entry must be `optional=True`,
    `status='not_installed'` -- not the generic red `not_ready`.
    """
    import src.routers.curation.models as models_mod

    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://segmenter:8000')

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == 'segmenter':
            return httpx.Response(200, json={'status': 'healthy', 'model': 'sam3', 'loaded': True})
        if request.url.path == '/v2/repository/index':
            # Repository index lists everything except the never-shipped
            # detector -- OCR det/rec, CLIP, PE encoder are all present.
            return httpx.Response(
                200,
                json=[
                    {'name': 'paddleocr_det_trt', 'version': '1', 'state': 'READY'},
                    {'name': 'paddleocr_rec_trt', 'version': '1', 'state': 'READY'},
                ],
            )
        if 'metrics' in request.url.path:
            return httpx.Response(200, text='')
        raise httpx.ConnectError('unexpected host', request=request)

    _patch_async_client(monkeypatch, models_mod, handler)

    result = await models_mod.models_status()
    models_by_name = {m['name']: m for m in result['models']}

    detector = models_by_name['license_plate_detector']
    assert detector['optional'] is True
    assert detector['status'] == 'not_installed'

    # A present-but-unloaded model (in the index but not READY) keeps the
    # unchanged 'not_ready' status -- confirms the new branch is scoped to
    # index-absence, not to "any non-ready state".
    ocr_rec_entry = models_by_name['paddleocr_rec_trt']
    assert ocr_rec_entry['status'] == 'ready'  # sanity: our mock lists it READY

    # Every non-detector, non-segmenter, non-VLM entry stays optional=False.
    vlm_name = models_mod._get_vlm_labeler().model
    for name, entry in models_by_name.items():
        if name in ('license_plate_detector', 'sam3', vlm_name):
            continue
        assert entry['optional'] is False


async def test_detector_present_but_unloaded_stays_not_ready(
    monkeypatch: pytest.MonkeyPatch,
    reference_region_profile: None,
) -> None:
    """Present-in-the-index-but-not-READY (e.g. UNAVAILABLE, or loading)
    is a real regression signal, not "not installed" -- must stay
    `not_ready` even though the detector is marked `optional`.
    """
    import src.routers.curation.models as models_mod

    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://segmenter:8000')

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == 'segmenter':
            return httpx.Response(200, json={'status': 'healthy', 'model': 'sam3', 'loaded': True})
        if request.url.path == '/v2/repository/index':
            return httpx.Response(
                200,
                json=[
                    {'name': 'license_plate_detector', 'version': '1', 'state': 'UNAVAILABLE'},
                ],
            )
        if 'metrics' in request.url.path:
            return httpx.Response(200, text='')
        raise httpx.ConnectError('unexpected host', request=request)

    _patch_async_client(monkeypatch, models_mod, handler)

    result = await models_mod.models_status()
    models_by_name = {m['name']: m for m in result['models']}

    detector = models_by_name['license_plate_detector']
    assert detector['optional'] is True
    assert detector['status'] == 'not_ready'


async def test_no_segmenter_configured_detector_not_optional(
    monkeypatch: pytest.MonkeyPatch,
    reference_region_profile: None,
) -> None:
    """Without a configured segmenter fallback, a missing detector is a
    real stall -- `optional` must be False and status falls back to the
    unchanged `not_ready` (matching the pre-existing behavior), even
    though the model is entirely absent from the repository index.
    """
    import dataclasses

    import src.routers.curation.models as models_mod
    from src.services.detection import profile_registry

    # Clear the active profile's segmenter_name so `models_status()` never
    # treats the detector as optional. `DetectionProfile` is a frozen
    # dataclass, so swap in a `dataclasses.replace()`'d copy behind the
    # module's own `get_active_region_profile` reference rather than
    # mutating the loaded instance in place.
    profile = profile_registry.get_active_region_profile()
    assert profile is not None
    no_segmenter_profile = dataclasses.replace(profile, segmenter_name='')
    monkeypatch.setattr(models_mod, 'get_active_region_profile', lambda: no_segmenter_profile)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == '/v2/repository/index':
            return httpx.Response(200, json=[])
        if 'metrics' in request.url.path:
            return httpx.Response(200, text='')
        raise httpx.ConnectError('unexpected host', request=request)

    _patch_async_client(monkeypatch, models_mod, handler)

    result = await models_mod.models_status()
    models_by_name = {m['name']: m for m in result['models']}

    detector = models_by_name['license_plate_detector']
    assert detector['optional'] is False
    assert detector['status'] == 'not_ready'
