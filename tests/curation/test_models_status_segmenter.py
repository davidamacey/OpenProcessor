"""`GET /curation/models/status` must probe the segmenter as its own HTTP
service, not as a Triton model.

Bug: the segmenter entry's name comes from
``DetectionProfile.segmenter_name`` (e.g. ``sam3``), but it lives at its
own HTTP service (``OP_SEGMENTER_URL``), never on Triton. The old roster
builder ran every entry through ``_build_triton_entry`` (looking it up in
Triton's ``/v2/repository/index``), so a healthy segmenter always
reported ``not_ready`` since it never appears in that index.

These tests exercise ``build_segmenter_entry``/``_segmenter_health``
(``src/routers/curation/_models_segmenter.py``, re-exported through
``models.py``) directly (unit) and the full ``models_status()`` roster
assembly (wiring), with all HTTP mocked via ``httpx.MockTransport`` — no
live services.
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
    """Route every ``httpx.AsyncClient(...)`` constructed inside
    ``models_mod`` through a ``MockTransport`` running ``handler``,
    regardless of the constructor args the module passes (e.g. ``timeout``).
    """

    class _FakeAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs.pop('transport', None)
            kwargs['transport'] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(models_mod.httpx, 'AsyncClient', _FakeAsyncClient)


async def test_segmenter_healthy_reports_ready_external(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.routers.curation.models as models_mod

    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://segmenter:8000')

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == '/health'
        return httpx.Response(
            200,
            json={
                'status': 'healthy',
                'model': 'sam3',
                'device': 'cuda:0',
                'loaded': True,
                'instances': 2,
            },
        )

    _patch_async_client(monkeypatch, models_mod, handler)

    entry = await models_mod.build_segmenter_entry(
        'sam3', 'Segmenter', 'role', 'Promptable segmentation'
    )

    assert entry['kind'] == 'external'
    assert entry['status'] == 'ready'
    assert entry['last_error'] is None
    assert entry['endpoint'] == 'http://segmenter:8000'
    assert entry['name'] == 'sam3'


async def test_segmenter_unreachable_reports_unavailable_with_last_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.routers.curation.models as models_mod

    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://segmenter:8000')

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('connection refused', request=request)

    _patch_async_client(monkeypatch, models_mod, handler)

    entry = await models_mod.build_segmenter_entry(
        'sam3', 'Segmenter', 'role', 'Promptable segmentation'
    )

    assert entry['kind'] == 'external'
    assert entry['status'] == 'unavailable'
    assert entry['last_error']
    assert 'connection refused' in entry['last_error']


async def test_segmenter_reachable_but_not_loaded_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.routers.curation.models as models_mod

    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://segmenter:8000')

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'status': 'loading', 'model': 'sam3', 'loaded': False})

    _patch_async_client(monkeypatch, models_mod, handler)

    entry = await models_mod.build_segmenter_entry(
        'sam3', 'Segmenter', 'role', 'Promptable segmentation'
    )

    assert entry['status'] == 'unavailable'
    assert entry['last_error']


async def test_no_url_configured_reports_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.routers.curation.models as models_mod

    monkeypatch.delenv('OP_SEGMENTER_URL', raising=False)

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not be called
        raise AssertionError('no HTTP call should be made when OP_SEGMENTER_URL is unset')

    _patch_async_client(monkeypatch, models_mod, handler)

    entry = await models_mod.build_segmenter_entry(
        'sam3', 'Segmenter', 'role', 'Promptable segmentation'
    )

    assert entry['kind'] == 'external'
    assert entry['status'] == 'not_configured'
    assert entry['endpoint'] == ''
    assert entry['last_error']


async def test_models_status_routes_segmenter_to_external_not_triton(
    monkeypatch: pytest.MonkeyPatch,
    reference_region_profile: None,
) -> None:
    """Full `models_status()` wiring: the segmenter entry must be built as
    `kind='external'` via its own `/health`, never as a Triton entry keyed
    off Triton's `/v2/repository/index` -- even though it's part of the
    `_core_models()` roster returned by `DetectionProfile`.

    Regression check for the bug: before the fix, the segmenter's name
    ('sam3', from license_plate.json) was run through `_build_triton_entry`
    and, since Triton never lists it, always came back `kind='triton'`,
    `status='not_ready'` regardless of the segmenter's real health.
    """
    import src.routers.curation.models as models_mod

    # license_plate.json's segmenter_name is 'sam3'.
    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://segmenter:8000')

    segmenter_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == 'segmenter':
            segmenter_calls.append(request.url.path)
            return httpx.Response(200, json={'status': 'healthy', 'model': 'sam3', 'loaded': True})
        # Triton's own repository-index / metrics calls -- simulate Triton
        # unreachable; irrelevant to this test, which only cares that the
        # segmenter entry doesn't come from this path.
        raise httpx.ConnectError('triton down', request=request)

    _patch_async_client(monkeypatch, models_mod, handler)

    result = await models_mod.models_status()

    models_by_name = {m['name']: m for m in result['models']}
    assert 'sam3' in models_by_name
    segmenter_entry = models_by_name['sam3']
    assert segmenter_entry['kind'] == 'external'
    assert segmenter_entry['status'] == 'ready'
    assert segmenter_entry['last_error'] is None
    assert segmenter_entry['unloadable'] is False
    # Exactly one /health probe -- the segmenter is never looked up by name
    # in Triton's /v2/repository/index (that single call happens for the
    # rest of the roster and never mentions the segmenter at all).
    assert segmenter_calls == ['/health']

    # The VLM entry is also `kind='external'` -- exclude it too so the
    # "everything else is Triton" check below isn't confused by it. With
    # no OP_VLM_URL configured its health probe fails fast (no network
    # call), so it's still 'unavailable' regardless.
    vlm_name = models_mod._get_vlm_labeler().model
    assert models_by_name[vlm_name]['kind'] == 'external'
    assert models_by_name[vlm_name]['unloadable'] is False

    # Every other roster entry (region detector, OCR det/rec, CLIP, PE
    # encoder) is a real Triton entry and, with Triton unreachable here,
    # reports 'unavailable' -- confirming the mock wiring actually
    # exercised the Triton path for non-segmenter models.
    other_entries = [m for m in result['models'] if m['name'] not in ('sam3', vlm_name)]
    assert other_entries
    assert all(m['kind'] == 'triton' for m in other_entries)
    assert all(m['status'] == 'unavailable' for m in other_entries)
    assert all(m['unloadable'] is True for m in other_entries)
