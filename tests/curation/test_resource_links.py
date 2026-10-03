"""Pins for the served ``resource_links`` (GET /settings): one typed list,
service URLs from the explicit override or the REQUEST host + service port,
API docs as path-relative entries, and a non-blocking cached reachability
probe."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import CurationConfig
from src.core.request_origin import RequestOriginMiddleware, parse_origin
from src.services import resource_links as rl


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    rl.reset_reachability_cache()


def _cfg(monkeypatch: pytest.MonkeyPatch, **kw: object) -> None:
    cfg = CurationConfig(**kw)  # type: ignore[arg-type]
    monkeypatch.setattr('src.config.get_curation_config', lambda: cfg)


def _app() -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestOriginMiddleware)

    @app.get('/links')
    def links() -> dict[str, str | None]:
        return {lk.id: lk.url for lk in rl.resource_links_from_config()}

    @app.get('/mlflow')
    def mlflow() -> dict[str, str | None]:
        return {'url': rl.service_url('mlflow')}

    return TestClient(app)


def _by_id(links: list[rl.ResourceLink]) -> dict[str, rl.ResourceLink]:
    return {link.id: link for link in links}


def test_ids_are_stable_and_ordered(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch)
    assert [lk.id for lk in rl.resource_links_from_config()] == [
        'swagger',
        'redoc',
        'openapi_json',
        'grafana',
        'prometheus',
        'opensearch_dashboards',
        'mlflow',
        'triton_metrics',
        'dcgm_metrics',
    ]


def test_docs_entries_are_relative_always_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch)
    links = _by_id(rl.resource_links_from_config())
    for id_, path in (('swagger', '/docs'), ('redoc', '/redoc'), ('openapi_json', '/openapi.json')):
        assert (links[id_].kind, links[id_].url, links[id_].status) == ('docs', path, 'configured')
        assert links[id_].reachable is None


def test_no_request_context_means_not_configured_never_localhost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg(monkeypatch)
    g = _by_id(rl.resource_links_from_config())['grafana']
    assert (g.kind, g.url, g.status, g.reachable) == ('service', None, 'not_configured', None)
    assert 'OP_GRAFANA_PORT' in g.hint


def test_lan_host_gets_its_own_host_and_service_port(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch, grafana_port=4955, mlflow_port=4959)
    body = _app().get('/links', headers={'Host': '10.10.10.20:5184'}).json()
    assert body['grafana'] == 'http://10.10.10.20:4955'
    assert body['mlflow'] == 'http://10.10.10.20:4959'
    assert body['triton_metrics'] == 'http://10.10.10.20:4602/metrics'
    assert body['swagger'] == '/docs'


def test_localhost_host_stays_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch)
    body = _app().get('/links', headers={'Host': 'localhost:4953'}).json()
    assert body['prometheus'] == 'http://localhost:4604'


def test_forwarded_host_and_proto_win(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch)
    headers = {
        'Host': 'api:8000',
        'X-Forwarded-Host': 'ops.example.com:8443, other',
        'X-Forwarded-Proto': 'https',
    }
    assert _app().get('/links', headers=headers).json()['grafana'] == 'https://ops.example.com:4605'


def test_explicit_override_is_verbatim_and_beats_request_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg(monkeypatch, grafana_url='https://grafana.example.com/g', mlflow_public_url='http://m:4')
    client = _app()
    assert client.get('/links', headers={'Host': '10.10.10.20:5184'}).json()['grafana'] == (
        'https://grafana.example.com/g'
    )
    assert client.get('/mlflow', headers={'Host': '10.10.10.20'}).json() == {'url': 'http://m:4'}


def test_mlflow_service_url_derives_from_request_host(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch, mlflow_port=4959)
    assert _app().get('/mlflow', headers={'Host': '10.10.10.20:5184'}).json() == {
        'url': 'http://10.10.10.20:4959'
    }


def test_port_zero_disables_the_derived_link(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch, grafana_port=0)
    assert _app().get('/links', headers={'Host': 'h:1'}).json()['grafana'] is None


@pytest.mark.parametrize('host', ['evil.com/x', 'a b', 'h"x', ''])
def test_malformed_host_is_rejected(host: str) -> None:
    assert parse_origin(host, None, 'http') is None


def test_ipv6_host_keeps_brackets() -> None:
    assert parse_origin('[::1]:4953', None, 'http') == ('http', '[::1]')


@pytest.mark.asyncio
async def test_refresh_populates_cache_for_enabled_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(
        monkeypatch,
        prometheus_port=0,
        dashboards_port=0,
        triton_metrics_port=0,
        dcgm_port=0,
    )
    seen: list[str] = []

    async def fake_probe(url: str) -> bool:
        seen.append(url)
        return True

    monkeypatch.setattr(rl, '_probe', fake_probe)
    await rl.refresh_reachability()
    assert sorted(seen) == ['http://curation-mlflow:5000/health', 'http://grafana:3000/api/health']


@pytest.mark.asyncio
async def test_reachable_is_served_per_link(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch, grafana_url='http://g:1', mlflow_public_url='http://m:4')

    async def fake_probe(url: str) -> bool:
        return 'grafana' in url

    monkeypatch.setattr(rl, '_probe', fake_probe)
    assert _by_id(rl.resource_links_from_config())['grafana'].reachable is None
    await rl.refresh_reachability()
    links = _by_id(rl.resource_links_from_config())
    assert links['grafana'].reachable is True
    assert links['mlflow'].reachable is False


@pytest.mark.asyncio
async def test_schedule_refresh_does_not_block_and_is_ttl_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg(
        monkeypatch,
        grafana_url='http://g:1',
        prometheus_port=0,
        dashboards_port=0,
        mlflow_port=0,
        triton_metrics_port=0,
        dcgm_port=0,
    )
    gate = asyncio.Event()
    calls = 0

    async def slow_probe(url: str) -> bool:
        nonlocal calls
        calls += 1
        await gate.wait()
        return True

    monkeypatch.setattr(rl, '_probe', slow_probe)
    rl.schedule_reachability_refresh()
    rl.schedule_reachability_refresh()  # in flight: no second task
    await asyncio.sleep(0)
    assert _by_id(rl.resource_links_from_config())['grafana'].reachable is None
    gate.set()
    await rl.wait_for_inflight_refresh()
    assert _by_id(rl.resource_links_from_config())['grafana'].reachable is True
    rl.schedule_reachability_refresh()  # fresh: within TTL
    await rl.wait_for_inflight_refresh()
    assert calls == 1


@pytest.mark.asyncio
async def test_probe_failure_is_false_not_raise() -> None:
    assert await rl._probe('http://127.0.0.1:1/x') is False
