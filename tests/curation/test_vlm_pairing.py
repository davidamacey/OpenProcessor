"""Pack <-> region profile <-> VLM endpoint pairing (W9.5): the checks
themselves, and that EVERY entry point that changes one side of the triple
runs them (the R5-1 lesson: a combined change is paired with what it will
become, not with what it replaces)."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from curation.conftest import ACTIVE, SCOPED, good_probe
from src.config import DetectionProfile
from src.services.config_store.vlm_pairing import _CALLS, context_breakdown, vlm_pairing_issues
from src.services.labeling.vlm_catalog import load_catalog
from src.services.labeling.vlm_endpoint_body import VlmEndpointBody
from src.services.labeling.vlm_endpoints import VlmEndpoint
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


ROOT = Path(__file__).resolve().parents[2]
#: a catalog entry whose text reading has been verified (the flag the check reads)
VERIFIED = next(e for e in load_catalog() if e.text_reading_verified)


def _endpoint(*, catalog_id: str | None = None, probe: Any = None, **body: Any) -> VlmEndpoint:
    fields = {'base_url': 'http://vlm:8000/v1', 'model': 'm', 'catalog_id': catalog_id, **body}
    return VlmEndpoint(
        name='pairing',
        source='stored',
        revision=1,
        body=VlmEndpointBody(**fields),
        etag='e',
        last_probe=probe,
    )


def _codes(issues: list[Any]) -> dict[str, str]:
    return {i.code: i.severity for i in issues}


TINY = good_probe(max_model_len=1200, image_tokens=400)
ROOMY = good_probe(max_model_len=131072, image_tokens=260)


# ---- the constants the estimate depends on ----------------------------------


def test_the_max_tokens_the_estimate_assumes_are_the_ones_the_labeler_sends() -> None:
    """``_CALLS`` mirrors the ``vlm_labeler_*.py`` operation modules; a change on
    either side that leaves the other behind would make the estimate silently wrong."""
    trees = [
        ast.parse((ROOT / f'src/services/labeling/vlm_labeler_{part}.py').read_text())
        for part in ('classify', 'verify', 'visibility', 'combined')
    ]
    sent: dict[str, int] = {}
    for node in (n for tree in trees for n in ast.walk(tree)):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    for key, value in zip(sub.keys, sub.values, strict=True):
                        if (
                            isinstance(key, ast.Constant)
                            and key.value == 'max_tokens'
                            and isinstance(value, ast.Constant)
                        ):
                            sent[node.name] = int(value.value)
    method_for_call = {
        'class_batch': '_label_chunk',
        'open_class_batch': '_label_chunk_open',
        'region_verify_batch': '_verify_region_chunk',
        'combined_batch': '_label_combined_chunk',
        'region_visible_batch': '_region_visible_chunk',
    }
    assert {call for call, *_ in _CALLS} == set(method_for_call)
    for call, _system, _user, max_tokens, _kind in _CALLS:
        assert sent[method_for_call[call]] == max_tokens, call


# ---- the checks --------------------------------------------------------------


def test_an_unprobed_endpoint_has_no_context_estimate() -> None:
    assert context_breakdown(_endpoint(), GENERIC_ITEM_PACK, ['a']) is None
    issues = vlm_pairing_issues(
        _endpoint(), GENERIC_ITEM_PACK, None, mode='activate', class_names=['a']
    )
    assert 'vlm_context_too_small' not in _codes(issues)


def test_the_context_estimate_grows_with_images_and_classes() -> None:
    small = context_breakdown(_endpoint(probe=ROOMY, max_images_per_call=1), GENERIC_ITEM_PACK, [])
    large = context_breakdown(_endpoint(probe=ROOMY, max_images_per_call=8), GENERIC_ITEM_PACK, [])
    many = context_breakdown(
        _endpoint(probe=ROOMY, max_images_per_call=1),
        GENERIC_ITEM_PACK,
        [f'class_{i}' for i in range(300)],
    )
    assert small is not None
    assert large is not None
    assert many is not None
    by_call = lambda rows: {r['call']: r['estimate'] for r in rows}  # noqa: E731
    assert by_call(large)['combined_batch'] - by_call(small)['combined_batch'] == 7 * 260
    assert by_call(many)['class_batch'] > by_call(small)['class_batch']


def test_open_calls_use_the_smaller_open_image_cap() -> None:
    rows = context_breakdown(
        _endpoint(probe=ROOMY, max_images_per_call=8, open_images_per_call=2),
        GENERIC_ITEM_PACK,
        [],
    )
    assert rows is not None
    by_call = {r['call']: r for r in rows}
    assert by_call['open_class_batch']['images_per_call'] == 2
    assert by_call['class_batch']['images_per_call'] == 8


@pytest.mark.parametrize(
    ('mode', 'severity'), [('activate', 'error'), ('validate', 'warning'), ('run', 'warning')]
)
def test_a_context_that_is_too_small_is_an_error_only_when_activating(
    mode: str, severity: str
) -> None:
    issues = vlm_pairing_issues(
        _endpoint(probe=TINY),
        GENERIC_ITEM_PACK,
        None,
        mode=mode,  # type: ignore[arg-type]
        class_names=['a', 'b'],
    )
    too_small = [i for i in issues if i.code == 'vlm_context_too_small']
    assert too_small
    assert {i.severity for i in too_small} == {severity}
    detail = too_small[0].detail
    assert detail['estimate'] > detail['max_model_len'] == 1200
    assert set(detail['breakdown']) == {'prompt_tokens', 'image_tokens_each', 'max_tokens'}
    assert all(i.bypassable for i in too_small)


def test_a_roomy_context_raises_nothing() -> None:
    issues = vlm_pairing_issues(
        _endpoint(probe=ROOMY), GENERIC_ITEM_PACK, None, mode='activate', class_names=['a']
    )
    assert 'vlm_context_too_small' not in _codes(issues)


def test_a_server_that_rejects_its_own_cap_is_an_error_in_every_mode() -> None:
    capped = good_probe(
        ok=False,
        issues=[{'code': 'vlm_max_images_exceeds_server', 'severity': 'error', 'message': 'x'}],
    )
    for mode in ('activate', 'validate', 'run'):
        issues = vlm_pairing_issues(_endpoint(probe=capped), None, None, mode=mode)  # type: ignore[arg-type]
        codes = _codes(issues)
        assert codes.get('vlm_max_images_exceeds_server') == 'error', mode
        assert not next(i for i in issues if i.code == 'vlm_max_images_exceeds_server').bypassable


def test_multi_box_and_text_reading_warnings_follow_the_catalog_flags() -> None:
    profile = replace(DetectionProfile(name='p'), max_regions_per_item=4, text_reader='vlm')
    unknown = vlm_pairing_issues(_endpoint(), None, profile, mode='activate')
    assert {'vlm_multi_box_unverified', 'vlm_reads_text_unverified'} <= set(_codes(unknown))
    assert set(_codes(unknown).values()) == {'warning'}

    verified = vlm_pairing_issues(_endpoint(catalog_id=VERIFIED.id), None, profile, mode='activate')
    assert 'vlm_reads_text_unverified' not in _codes(verified)

    single = replace(profile, max_regions_per_item=1, text_reader='none')
    quiet = vlm_pairing_issues(_endpoint(), None, single, mode='activate')
    assert not {'vlm_multi_box_unverified', 'vlm_reads_text_unverified'} & set(_codes(quiet))


def test_json_mode_off_with_a_reasoning_channel_warns() -> None:
    probe = good_probe(reasoning_channel=True, json_mode_supported=False)
    issues = vlm_pairing_issues(_endpoint(probe=probe), None, None, mode='validate')
    assert _codes(issues).get('vlm_json_mode_off') == 'warning'
    on = vlm_pairing_issues(
        _endpoint(probe=good_probe(reasoning_channel=True)), None, None, mode='validate'
    )
    assert 'vlm_json_mode_off' not in _codes(on)


def test_open_images_larger_than_the_cap_is_flagged() -> None:
    issues = vlm_pairing_issues(
        _endpoint(max_images_per_call=2, open_images_per_call=6), None, None, mode='validate'
    )
    assert _codes(issues).get('vlm_open_images_clamped') == 'warning'


# ---- every entry point runs them ----------------------------------------------


def _activate_tiny_vlm(api) -> None:
    api.probe_record[0] = TINY
    api.ready('tiny')
    forced = api.activate('tiny', expected_active=None, force=True)
    assert forced.status_code == 200, forced.text


def _pack_body() -> dict[str, Any]:
    body = GENERIC_ITEM_PACK.to_dict()
    body.pop('name')
    return body


def _stored_pack(api) -> None:
    created = api.client.post(
        f'{SCOPED}/prompt_packs', json={'name': 'mypack', 'body': _pack_body()}
    )
    assert created.status_code == 201, created.text


def _too_small(response: Any) -> bool:
    return response.status_code == 422 and 'vlm_context_too_small' in response.text


def test_entry_point_activating_the_vlm(vlm_api) -> None:
    vlm_api.probe_record[0] = TINY
    vlm_api.ready('tiny')
    refused = vlm_api.activate('tiny', expected_active=None)
    assert _too_small(refused), refused.text
    detail = refused.json()['detail']['report']
    assert detail['force_allowed'] is True
    assert vlm_api.activate('tiny', expected_active=None, force=True).status_code == 200


def test_entry_point_activating_a_pack_against_the_active_vlm(vlm_api) -> None:
    _activate_tiny_vlm(vlm_api)
    _stored_pack(vlm_api)
    refused = vlm_api.client.post(
        f'{SCOPED}/prompt_packs/mypack/activate', json={'expected_active': None}
    )
    assert _too_small(refused), refused.text
    ok = vlm_api.client.post(
        f'{SCOPED}/prompt_packs/mypack/activate', json={'expected_active': None, 'force': True}
    )
    assert ok.status_code == 200, ok.text


def test_entry_point_a_combined_settings_change_pairs_the_pack_with_the_pending_vlm(
    vlm_api,
) -> None:
    """The stored VLM is roomy; the same request switches to a tiny one. The
    pack must be paired with the tiny one it is about to run against."""
    vlm_api.probe_record[0] = ROOMY
    vlm_api.ready('roomy')
    assert vlm_api.activate('roomy', expected_active=None).status_code == 200
    _stored_pack(vlm_api)
    vlm_api.probe_record[0] = TINY
    vlm_api.ready('tiny')

    response = vlm_api.client.put(
        f'{SCOPED}/settings', json={'defaults': {'vlm': 'tiny', 'prompt_pack': 'mypack'}}
    )
    assert _too_small(response), response.text
    assert vlm_api.active()['active']['name'] == 'roomy'


def test_entry_point_a_combined_change_that_switches_the_vlm_off_pairs_with_nothing(
    vlm_api,
) -> None:
    """The opposite direction: the stored VLM is too small for the pack, but
    the same request turns the VLM off, so the pack is not held to it."""
    _activate_tiny_vlm(vlm_api)
    _stored_pack(vlm_api)
    alone = vlm_api.client.put(f'{SCOPED}/settings', json={'defaults': {'prompt_pack': 'mypack'}})
    assert _too_small(alone), alone.text

    combined = vlm_api.client.put(
        f'{SCOPED}/settings', json={'defaults': {'vlm': 'off', 'prompt_pack': 'mypack'}}
    )
    assert combined.status_code == 200, combined.text
    assert vlm_api.active()['active'] == {'name': None, 'revision': None}


def test_entry_point_per_run_selection_only_warns_about_context(vlm_api) -> None:
    """A per-run selection carries no ``force``, so an estimate is a warning
    there (only the certain failure blocks)."""
    from src.routers.curation.pipeline_vlm import resolve_run_vlm

    _stored_pack(vlm_api)
    vlm_api.probe_record[0] = TINY
    vlm_api.ready('tiny')

    import asyncio

    run = asyncio.run(resolve_run_vlm(vlm_api.fake_os, 'tiny', pack=None))
    assert (run.name, run.revision) == ('tiny', 1)


def test_active_vlm_pairing_hook_reads_the_active_endpoint(vlm_api) -> None:
    """The hook pack and profile activation call: the pairing issues of the
    project's active VLM with the (pack, profile) about to be activated."""
    import asyncio

    from src.services.config_store.vlm_gate import active_vlm_pairing_issues

    vlm_api.probe_record[0] = TINY
    vlm_api.ready('tiny')
    assert vlm_api.activate('tiny', expected_active=None, force=True).status_code == 200
    issues = asyncio.run(active_vlm_pairing_issues(vlm_api.fake_os, GENERIC_ITEM_PACK, None))
    assert 'vlm_context_too_small' in [i.code for i in issues]

    # explicit "the VLM is being switched off in this request" pairs with nothing
    off = asyncio.run(
        active_vlm_pairing_issues(vlm_api.fake_os, GENERIC_ITEM_PACK, None, pending_vlm=None)
    )
    assert off == []
    assert ACTIVE.endswith('/vlm/endpoints')


@pytest.mark.parametrize('entry', load_catalog(), ids=lambda e: e.id)
def test_every_catalog_entry_serves_a_context_its_own_image_cap_fits(entry: Any) -> None:
    """Found live: the shipped default (8192 tokens, 8 images per call) failed
    this activation check for the very first pack and profile, so the wheel
    example could not be activated without ``force``."""
    probe = good_probe(max_model_len=entry.max_model_len, image_tokens=267)
    endpoint = _endpoint(catalog_id=entry.id, probe=probe, max_images_per_call=entry.max_images)

    issues = vlm_pairing_issues(
        endpoint, GENERIC_ITEM_PACK, None, mode='activate', class_names=['car', 'wheel']
    )

    assert 'vlm_context_too_small' not in _codes(issues)
