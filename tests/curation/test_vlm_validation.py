"""``validate_vlm_endpoint``: the one validator behind validate, create,
clone, PUT, activate and per-run resolution (W9.4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import src.services.labeling.vlm_url_policy as policy
from src.services.config_store.vlm_validation import (
    BYPASSABLE_CODES,
    external_policy,
    validate_vlm_endpoint,
)
from src.services.labeling.vlm_catalog import load_catalog
from src.services.labeling.vlm_endpoint_body import VlmEndpointBody, VlmProbeRecord


if TYPE_CHECKING:
    from pathlib import Path

PROBED_AT = '2026-09-28T12:00:00+00:00'


@pytest.fixture(autouse=True)
def _dns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    table = {
        'vlm': ['172.18.0.9'],
        'gpu.lan': ['192.168.1.20'],
        'api.example.com': ['93.184.216.34'],
    }
    monkeypatch.setattr(policy, '_resolve', lambda host: list(table.get(host, [])))
    monkeypatch.setattr(policy, '_docker_gateway_addresses', lambda: frozenset())
    policy.reset_policy_caches()
    monkeypatch.setenv('OP_VLM_SECRETS_DIR', str(tmp_path / 'secrets'))
    monkeypatch.delenv('OP_VLM_EXTERNAL_POLICY', raising=False)


def _body(**over: object) -> VlmEndpointBody:
    fields: dict[str, object] = {'base_url': 'http://vlm:8000/v1', 'model': 'm'}
    fields.update(over)
    return VlmEndpointBody(**fields)  # type: ignore[arg-type]


def _probe(*, errors: list[dict[str, object]] | None = None, ok: bool = True) -> VlmProbeRecord:
    return VlmProbeRecord(ok=ok, probed_at=PROBED_AT, issues=errors or [])


def _codes(report) -> set[str]:
    return {i.code for i in [*report.errors, *report.warnings]}


async def _validate(body: VlmEndpointBody, **kw: object):
    kw.setdefault('name', 'my-vlm')
    kw.setdefault('probe', None)
    kw.setdefault('for_activation', False)
    return await validate_vlm_endpoint(body, **kw)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_plain_local_endpoint_is_valid() -> None:
    report, locality = await _validate(_body())
    assert report.ok
    assert report.errors == []
    assert locality == 'compose'


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['a', 'A-upper', '-lead', 'has space', 'x' * 65, 'ok/slash', ''])
async def test_bad_names_are_refused(name: str) -> None:
    report, _ = await _validate(_body(), name=name)
    assert 'vlm_name_invalid' in _codes(report)
    assert not report.ok


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['env', 'off', 'none', 'default', 'local', 'active', 'schema'])
async def test_reserved_names_are_refused(name: str) -> None:
    report, _ = await _validate(_body(), name=name)
    assert 'vlm_name_reserved' in _codes(report)


@pytest.mark.asyncio
async def test_an_existing_name_conflicts_only_when_names_are_supplied() -> None:
    clash, _ = await _validate(_body(), existing_names={'my-vlm'})
    assert 'name_conflict' in _codes(clash)
    fine, _ = await _validate(_body())
    assert 'name_conflict' not in _codes(fine)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('max_images_per_call', 0),
        ('max_images_per_call', 10_000),
        ('open_images_per_call', 0),
        ('timeout_s', 0.0),
        ('timeout_s', 1e9),
        ('requests_per_second', 0.0),
        ('model', '   '),
    ],
)
async def test_out_of_range_fields_are_errors(field: str, value: object) -> None:
    report, _ = await _validate(_body(**{field: value}))
    assert 'vlm_field_range' in _codes(report)
    assert any(e.field == field for e in report.errors)


@pytest.mark.asyncio
async def test_a_bad_url_is_an_error_with_no_locality() -> None:
    report, locality = await _validate(_body(base_url='ftp://vlm/v1'))
    assert 'vlm_url_invalid' in _codes(report)
    assert locality is None


@pytest.mark.asyncio
async def test_ssrf_targets_are_errors_and_are_never_bypassable() -> None:
    for url in ('http://169.254.169.254/v1', 'http://opensearch:9200/v1', 'http://[::]/v1'):
        report, locality = await _validate(_body(base_url=url), for_activation=True)
        assert not report.ok, url
        assert locality is None
        assert report.force_allowed is False
        denials = [e for e in report.errors if e.code.startswith('vlm_url_denied')]
        assert denials
        assert not any(e.bypassable for e in denials)


@pytest.mark.asyncio
async def test_external_endpoint_needs_the_acknowledgement_flag() -> None:
    url = 'https://api.example.com/v1'
    refused, locality = await _validate(_body(base_url=url))
    assert locality == 'external'
    assert 'vlm_external_not_acknowledged' in _codes(refused)
    ok, _ = await _validate(_body(base_url=url, allow_external=True))
    assert ok.ok


@pytest.mark.asyncio
async def test_an_unresolvable_host_counts_as_external() -> None:
    report, locality = await _validate(_body(base_url='http://nowhere.example/v1'))
    assert locality == 'unknown'
    assert 'vlm_external_not_acknowledged' in _codes(report)


@pytest.mark.asyncio
async def test_deny_policy_refuses_external_even_when_acknowledged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_VLM_EXTERNAL_POLICY', 'deny')
    report, _ = await _validate(_body(base_url='https://api.example.com/v1', allow_external=True))
    assert 'vlm_external_denied' in _codes(report)
    assert not report.ok
    # a local endpoint is unaffected
    assert (await _validate(_body()))[0].ok


@pytest.mark.parametrize('raw', ['DENY', 'block', 'ack ', 'nonsense'])
def test_an_unrecognised_policy_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv('OP_VLM_EXTERNAL_POLICY', raw)
    assert external_policy() == ('ack' if raw.strip().lower() == 'ack' else 'deny')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'ref',
    ['sk-live-abc123', 'env:HOME', 'env:OP_VLM_API_KEY', 'secret:../etc/passwd', 'secret:A B'],
)
async def test_api_key_refs_must_be_secret_references(ref: str) -> None:
    report, _ = await _validate(_body(api_key_ref=ref))
    assert 'vlm_api_key_ref_invalid' in _codes(report)
    assert not report.ok


@pytest.mark.asyncio
async def test_the_env_key_reference_is_valid_only_on_the_env_builtin() -> None:
    stored, _ = await _validate(_body(api_key_ref='env:OP_VLM_API_KEY'))
    assert 'vlm_api_key_ref_invalid' in _codes(stored)
    builtin, _ = await _validate(_body(api_key_ref='env:OP_VLM_API_KEY'), is_env=True)
    assert 'vlm_api_key_ref_invalid' not in _codes(builtin)


@pytest.mark.asyncio
async def test_a_missing_secret_warns_on_save_and_errors_on_activation(tmp_path: Path) -> None:
    body = _body(api_key_ref='secret:vendor')
    saved, _ = await _validate(body)
    assert saved.ok
    assert 'vlm_api_key_unresolved' in {w.code for w in saved.warnings}
    activating, _ = await _validate(body, for_activation=True, probe=_probe())
    assert 'vlm_api_key_unresolved' in {e.code for e in activating.errors}

    secrets = tmp_path / 'secrets'
    secrets.mkdir()
    (secrets / 'vendor').write_text('k')
    present, _ = await _validate(body, for_activation=True, probe=_probe())
    assert 'vlm_api_key_unresolved' not in _codes(present)


@pytest.mark.asyncio
async def test_an_unknown_catalog_id_is_only_a_warning() -> None:
    report, _ = await _validate(_body(catalog_id='no-such-model'))
    assert report.ok
    assert 'vlm_catalog_id_unknown' in {w.code for w in report.warnings}
    known, _ = await _validate(_body(catalog_id=load_catalog()[0].id))
    assert 'vlm_catalog_id_unknown' not in _codes(known)


@pytest.mark.asyncio
async def test_activation_needs_a_probe_but_force_may_bypass_it() -> None:
    report, _ = await _validate(_body(), for_activation=True)
    assert [e.code for e in report.errors] == ['vlm_not_probed']
    assert report.force_allowed is True
    assert report.errors[0].bypassable is True
    # the env built-in is exempt: nothing to probe before first use
    builtin, _ = await _validate(_body(), for_activation=True, is_env=True)
    assert builtin.ok


@pytest.mark.asyncio
async def test_a_failed_probe_blocks_activation_bypassably() -> None:
    failed = _probe(
        ok=False,
        errors=[{'code': 'vlm_model_not_served', 'severity': 'error', 'message': 'not served'}],
    )
    report, _ = await _validate(_body(), for_activation=True, probe=failed)
    assert [e.code for e in report.errors] == ['vlm_probe_failed']
    assert report.force_allowed is True


@pytest.mark.asyncio
async def test_the_servers_image_cap_error_is_never_bypassable() -> None:
    capped = _probe(
        ok=False,
        errors=[
            {
                'code': 'vlm_max_images_exceeds_server',
                'severity': 'error',
                'message': 'the server takes 2 images',
                'detail': {'server_max': 2},
            }
        ],
    )
    report, _ = await _validate(_body(), for_activation=True, probe=capped)
    assert [e.code for e in report.errors] == ['vlm_max_images_exceeds_server']
    assert report.force_allowed is False
    assert 'vlm_max_images_exceeds_server' not in BYPASSABLE_CODES
