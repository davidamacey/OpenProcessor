"""Pins for the served ``resource_links`` (GET /settings): one typed list,
service URLs from the explicit override or the REQUEST host + service port,
API docs as path-relative entries, and a non-blocking cached reachability
probe."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import CurationConfig
from src.core.request_origin import RequestOriginMiddleware, parse_origin
from src.services import resource_links as rl


def _cfg(monkeypatch: pytest.MonkeyPatch, **kw: object) -> None:
    cfg = CurationConfig(**kw)  # type: ignore[arg-type]
    monkeypatch.setattr('src.config.get_curation_config', lambda: cfg)


def _app() -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestOriginMiddleware)

    @app.get('/links')
    def links() -> dict[str, str | None]:
        return {lk.id: lk.url for lk in rl.resource_links_from_config({})}

    @app.get('/mlflow')
    def mlflow() -> dict[str, str | None]:
        return {'url': rl.service_url('mlflow')}

    return TestClient(app)


def _by_id(links: list[rl.ResourceLink]) -> dict[str, rl.ResourceLink]:
    return {link.id: link for link in links}


def test_ids_are_stable_and_ordered(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch)
    assert [lk.id for lk in rl.resource_links_from_config({})] == [
        'swagger',
        'redoc',
        'openapi_json',
        'grafana',
        'prometheus',
        'opensearch_dashboards',
        'mlflow',
    ]


def test_docs_entries_are_relative_always_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch)
    links = _by_id(rl.resource_links_from_config({}))
    for id_, path in (('swagger', '/docs'), ('redoc', '/redoc'), ('openapi_json', '/openapi.json')):
        assert (links[id_].kind, links[id_].url, links[id_].status) == ('docs', path, 'configured')
        assert links[id_].reachable is None


def test_no_request_context_means_not_configured_never_localhost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg(monkeypatch)
    g = _by_id(rl.resource_links_from_config({}))['grafana']
    assert (g.kind, g.url, g.status, g.reachable) == ('service', None, 'not_configured', None)
    assert 'OP_GRAFANA_PORT' in g.hint


def test_lan_host_gets_its_own_host_and_service_port(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch, grafana_port=4955, mlflow_port=4959)
    body = _app().get('/links', headers={'Host': '192.0.2.10:5184'}).json()
    assert body['grafana'] == 'http://192.0.2.10:4955'
    assert body['mlflow'] == 'http://192.0.2.10:4959'
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
    assert client.get('/links', headers={'Host': '192.0.2.10:5184'}).json()['grafana'] == (
        'https://grafana.example.com/g'
    )
    assert client.get('/mlflow', headers={'Host': '192.0.2.10'}).json() == {'url': 'http://m:4'}


def test_mlflow_service_url_derives_from_request_host(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch, mlflow_port=4959)
    assert _app().get('/mlflow', headers={'Host': '192.0.2.10:5184'}).json() == {
        'url': 'http://192.0.2.10:4959'
    }


def test_port_zero_disables_the_derived_link(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch, grafana_port=0)
    assert _app().get('/links', headers={'Host': 'h:1'}).json()['grafana'] is None


@pytest.mark.parametrize('host', ['evil.com/x', 'a b', 'h"x', ''])
def test_malformed_host_is_rejected(host: str) -> None:
    assert parse_origin(host, None, 'http') is None


def test_ipv6_host_keeps_brackets() -> None:
    assert parse_origin('[::1]:4953', None, 'http') == ('http', '[::1]')


def test_reachable_is_served_per_link_from_the_published_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg(monkeypatch, grafana_url='http://g:1', mlflow_public_url='http://m:4')
    assert _by_id(rl.resource_links_from_config({}))['grafana'].reachable is None
    links = _by_id(rl.resource_links_from_config({'grafana': True, 'mlflow': False}))
    assert links['grafana'].reachable is True
    assert links['mlflow'].reachable is False
    assert links['prometheus'].reachable is None


# --- gateway mode (OP_GATEWAY_SUBPATHS): path-relative links ---------------

_GATEWAY_PATHS = {
    'grafana': '/grafana/',
    'prometheus': '/prometheus/',
    'opensearch_dashboards': '/dashboards/',
    'mlflow': '/mlflow/',
}


def test_gateway_mode_serves_path_relative_links(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch, gateway_subpaths=True)
    body = _app().get('/links', headers={'Host': '192.0.2.10:5184'}).json()
    assert {k: body[k] for k in _GATEWAY_PATHS} == _GATEWAY_PATHS
    assert body['swagger'] == '/docs'


def test_gateway_mode_needs_no_request_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch, gateway_subpaths=True)
    links = _by_id(rl.resource_links_from_config({}))
    for id_, path in _GATEWAY_PATHS.items():
        assert (links[id_].url, links[id_].status) == (path, 'configured')
    assert rl.service_url('mlflow') == '/mlflow/'


def test_gateway_off_keeps_host_port_derivation(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch, gateway_subpaths=False)
    assert _app().get('/links', headers={'Host': '192.0.2.10:5184'}).json()['grafana'] == (
        'http://192.0.2.10:4605'
    )
    assert rl.service_url('grafana') is None


def test_gateway_mode_explicit_override_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(
        monkeypatch,
        gateway_subpaths=True,
        grafana_url='https://grafana.example.com/g',
        mlflow_public_url='http://m:4',
    )
    links = _by_id(rl.resource_links_from_config({}))
    assert links['grafana'].url == 'https://grafana.example.com/g'
    assert links['mlflow'].url == 'http://m:4'
    assert links['prometheus'].url == '/prometheus/'


def test_gateway_mode_port_zero_still_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch, gateway_subpaths=True, grafana_port=0)
    links = _by_id(rl.resource_links_from_config({}))
    assert (links['grafana'].url, links['grafana'].status) == (None, 'not_configured')
    assert links['mlflow'].url == '/mlflow/'


def test_gateway_mode_port_zero_with_explicit_url_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg(monkeypatch, gateway_subpaths=True, mlflow_port=0, mlflow_public_url='http://m:4')
    assert rl.service_url('mlflow') == 'http://m:4'


def test_gateway_mode_hint_names_the_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch, gateway_subpaths=True)
    assert 'gateway' in _by_id(rl.resource_links_from_config({}))['grafana'].hint.lower()


def test_mlflow_run_url_follows_the_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.training.job_wire import _public_mlflow_url

    _cfg(monkeypatch, gateway_subpaths=True)
    assert _public_mlflow_url('run1', '7') == '/mlflow/#/experiments/7/runs/run1'


def test_gateway_flag_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert CurationConfig.from_env().gateway_subpaths is False
    monkeypatch.setenv('OP_GATEWAY_SUBPATHS', 'true')
    assert CurationConfig.from_env().gateway_subpaths is True
    monkeypatch.setenv('OP_GATEWAY_SUBPATHS', 'false')
    assert CurationConfig.from_env().gateway_subpaths is False
