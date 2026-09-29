"""``labeler_for`` / ``build_uncached_labeler`` (W9.0): the one place a
labeler is built. What it applies from the endpoint, when it rebuilds, and
that it re-checks the SSRF policy at construction."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

import src.services.labeling.vlm_url_policy as policy
from src.services.labeling import vlm_factory
from src.services.labeling.vlm_client import VlmIdentity
from src.services.labeling.vlm_endpoint_body import VlmEndpointBody, VlmProbeRecord
from src.services.labeling.vlm_endpoints import VlmEndpoint, VlmEndpointUnavailableError
from src.services.labeling.vlm_factory import (
    VlmEndpointDeniedError,
    build_uncached_labeler,
    labeler_for,
    reset_labeler_cache,
)
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, PromptPack


@pytest.fixture(autouse=True)
def _dns(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    table = {'vlm': ['172.18.0.9'], 'sneaky.example.com': ['93.184.216.34']}
    monkeypatch.setattr(policy, '_resolve', lambda host: list(table.get(host, [])))
    monkeypatch.setenv('OP_VLM_SECRETS_DIR', str(tmp_path))
    policy.reset_policy_caches()
    reset_labeler_cache()


def _endpoint(
    *, revision: int | None = 1, probe: VlmProbeRecord | None = None, **body: Any
) -> VlmEndpoint:
    fields = {'base_url': 'http://vlm:8000/v1', 'model': 'm', **body}
    return VlmEndpoint(
        name='one',
        source='stored',
        revision=revision,
        body=VlmEndpointBody(**fields),
        etag='e',
        last_probe=probe,
    )


def _probe(**over: Any) -> VlmProbeRecord:
    return VlmProbeRecord(ok=True, probed_at='2026-09-28T12:00:00+00:00', **over)


def test_the_labeler_takes_everything_from_the_endpoint() -> None:
    endpoint = _endpoint(
        max_images_per_call=3,
        open_images_per_call=2,
        timeout_s=33.0,
        requests_per_second=7.0,
        json_mode='off',
        probe=_probe(root='org/real'),
    )
    labeler = labeler_for(endpoint, GENERIC_ITEM_PACK)
    assert labeler.base_url == 'http://vlm:8000/v1'
    assert labeler.model == 'm'  # what is SENT is the served alias
    assert labeler.max_images_per_call == 3
    assert labeler.open_images_per_call == 2
    assert labeler.timeout_s == 33.0
    assert labeler.requests_per_second == 7.0
    assert labeler.json_mode is False
    assert labeler.identity == VlmIdentity('one@1', 'org/real')  # what is RECORDED is the root
    assert labeler._pack is GENERIC_ITEM_PACK


def test_the_endpoints_own_cap_is_the_ceiling_not_the_deployment_default() -> None:
    """The deployment default (``OP_VLM_MAX_IMAGES_PER_CALL``, 8) used to
    clamp every labeler; an endpoint that takes 12 images per call gets 12."""
    labeler = labeler_for(_endpoint(max_images_per_call=12), GENERIC_ITEM_PACK)
    assert labeler.max_images_per_call == 12


def test_json_mode_auto_follows_the_probe() -> None:
    assert labeler_for(_endpoint(), GENERIC_ITEM_PACK).json_mode is True
    reset_labeler_cache()
    off = _endpoint(probe=_probe(json_mode_supported=False))
    assert labeler_for(off, GENERIC_ITEM_PACK).json_mode is False


def test_equal_inputs_share_one_labeler() -> None:
    a = labeler_for(_endpoint(), GENERIC_ITEM_PACK)
    b = labeler_for(_endpoint(), GENERIC_ITEM_PACK)
    assert a is b


def _cases() -> list[tuple[str, VlmEndpoint, PromptPack]]:
    other_pack = replace(GENERIC_ITEM_PACK, name='another_pack')
    edited_pack = replace(
        GENERIC_ITEM_PACK, class_system=GENERIC_ITEM_PACK.class_system + ' edited'
    )
    return [
        ('revision', _endpoint(revision=2), GENERIC_ITEM_PACK),
        ('pack name', _endpoint(), other_pack),
        ('pack content', _endpoint(), edited_pack),
        ('json mode', _endpoint(json_mode='off'), GENERIC_ITEM_PACK),
        ('probe', _endpoint(probe=_probe(root='org/x')), GENERIC_ITEM_PACK),
        (
            'resolved revision tag',
            _endpoint(),
            _tagged(GENERIC_ITEM_PACK, 7),
        ),
    ]


def _tagged(pack: PromptPack, revision: int) -> PromptPack:
    tagged = replace(pack)
    object.__setattr__(tagged, '_resolved_revision', revision)
    return tagged


@pytest.mark.parametrize('case', range(6))
def test_anything_the_labeler_is_built_from_makes_a_new_one(case: int) -> None:
    label, endpoint, pack = _cases()[case]
    base = labeler_for(_endpoint(), GENERIC_ITEM_PACK)
    assert labeler_for(endpoint, pack) is not base, label


def test_a_reprobe_is_a_new_labeler_even_when_the_facts_are_the_same() -> None:
    """The probe marker changes with the probe's timestamp: that is how a
    rotated key file is picked up."""
    first = labeler_for(_endpoint(probe=_probe()), GENERIC_ITEM_PACK)
    later = _probe()
    later = later.model_copy(update={'probed_at': '2026-09-28T13:00:00+00:00'})
    assert labeler_for(_endpoint(probe=later), GENERIC_ITEM_PACK) is not first


def test_the_cache_is_bounded_and_defers_closing_what_it_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[Any] = []

    def defer(labeler: Any) -> None:
        closed.append(labeler)

    monkeypatch.setattr(vlm_factory, '_schedule_close', defer)
    built = [labeler_for(_endpoint(revision=n), GENERIC_ITEM_PACK) for n in range(1, 40)]
    assert len(vlm_factory._LABELERS) == vlm_factory._CACHE_MAX
    assert (
        closed == built[: len(built) - vlm_factory._CACHE_MAX]
    )  # oldest first, never closed inline


@pytest.mark.asyncio
async def test_a_replaced_labeler_is_closed_only_after_in_flight_calls_can_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waited: list[float] = []
    closed: list[str] = []

    async def fake_sleep(seconds: float) -> None:
        waited.append(seconds)

    class Fake:
        timeout_s = 20.0

        async def aclose(self) -> None:
            closed.append('closed')

    monkeypatch.setattr(vlm_factory.asyncio, 'sleep', fake_sleep)
    vlm_factory._schedule_close(Fake())  # type: ignore[arg-type]
    await asyncio.gather(*list(vlm_factory._CLOSING))
    assert waited == [20.0 + vlm_factory._CLOSE_GRACE_S]
    assert closed == ['closed']


def test_the_uncached_builder_owns_what_it_returns() -> None:
    a = build_uncached_labeler(_endpoint(), GENERIC_ITEM_PACK)
    b = build_uncached_labeler(_endpoint(), GENERIC_ITEM_PACK)
    assert a is not b
    assert vlm_factory._LABELERS == {} or a not in vlm_factory._LABELERS.values()


def test_a_url_that_turned_forbidden_after_it_was_saved_is_refused_at_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = _endpoint(base_url='http://sneaky.example.com/v1')
    assert labeler_for(good, GENERIC_ITEM_PACK) is not None
    reset_labeler_cache()
    monkeypatch.setattr(policy, '_resolve', lambda _host: ['169.254.169.254'])  # rebinding
    policy.reset_policy_caches()
    with pytest.raises(VlmEndpointDeniedError):
        labeler_for(good, GENERIC_ITEM_PACK)
    with pytest.raises(VlmEndpointDeniedError):
        build_uncached_labeler(good, GENERIC_ITEM_PACK)
    assert issubclass(VlmEndpointDeniedError, VlmEndpointUnavailableError)


def test_the_env_builtin_is_not_subject_to_the_url_denial(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator's own setting (for example a loopback vLLM on the host)
    is theirs to make; only stored endpoints are policed."""
    endpoint = replace(
        _endpoint(base_url='http://127.0.0.1:8000/v1'), source='env', revision=None, name='env'
    )
    assert labeler_for(endpoint, GENERIC_ITEM_PACK) is not None


def test_a_stored_endpoints_missing_secret_fails_closed_not_open(tmp_path: Any) -> None:
    with pytest.raises(VlmEndpointUnavailableError, match='missing or empty'):
        labeler_for(_endpoint(api_key_ref='secret:absent'), GENERIC_ITEM_PACK)
    (tmp_path / 'present').write_text('k')
    labeler = labeler_for(_endpoint(api_key_ref='secret:present'), GENERIC_ITEM_PACK)
    assert labeler.api_key == 'k'


def test_no_key_means_the_conventional_placeholder() -> None:
    assert labeler_for(_endpoint(), GENERIC_ITEM_PACK).api_key == 'EMPTY'
