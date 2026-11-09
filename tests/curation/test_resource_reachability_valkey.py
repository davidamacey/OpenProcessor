"""``resource_links`` reachability is one answer for every API process.

The beat tick ``op.periodic.resource_reachability`` probes the services once per deployment and
writes ``op:resource_reachability`` to Valkey with a 90 s expiry; every API process only reads it.
A missing key (the tick has not run, or a Valkey outage) reads ``unknown``, never ``down``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fixtures.fake_valkey import FakeValkey

from src.config import CurationConfig
from src.services import resource_links as rl


def _cfg(monkeypatch: pytest.MonkeyPatch, **kw: object) -> None:
    cfg = CurationConfig(**kw)  # type: ignore[arg-type]
    monkeypatch.setattr('src.config.get_curation_config', lambda: cfg)


@pytest.mark.asyncio
async def test_the_tick_probes_the_enabled_services_and_writes_one_key_for_90_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.periodic import resource_reachability

    _cfg(monkeypatch, prometheus_port=0, dashboards_port=0)
    shared = FakeValkey()
    monkeypatch.setattr(rl, 'valkey_client', lambda: shared)
    seen: list[str] = []

    async def fake_probe(url: str) -> bool:
        seen.append(url)
        return 'grafana' in url

    monkeypatch.setattr(rl, '_probe', fake_probe)

    await resource_reachability.tick()

    assert sorted(seen) == ['http://curation-mlflow:5000/health', 'http://grafana:3000/api/health']
    value, ttl = shared.data['op:resource_reachability']
    assert (json.loads(value), ttl) == ({'grafana': True, 'mlflow': False}, 90)


@pytest.mark.asyncio
async def test_two_api_processes_read_the_same_answer_from_one_valkey(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg(monkeypatch, grafana_url='http://g:1', mlflow_public_url='http://m:4')
    shared = FakeValkey()
    await rl.publish_reachability({'grafana': True, 'mlflow': False}, valkey=shared)

    answers: list[dict[str, bool | None]] = []
    for _process in range(2):
        monkeypatch.setattr(rl, 'valkey_client', lambda: FakeValkey(shared.data))
        links = rl.resource_links_from_config(await rl.read_reachability())
        answers.append({link.id: link.reachable for link in links if link.kind == 'service'})

    assert answers[0] == answers[1]
    assert answers[0]['grafana'] is True
    assert answers[0]['mlflow'] is False


@pytest.mark.asyncio
async def test_a_missing_key_is_unknown_not_down(monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg(monkeypatch, grafana_url='http://g:1', mlflow_public_url='http://m:4')
    monkeypatch.setattr(rl, 'valkey_client', FakeValkey)

    links = rl.resource_links_from_config(await rl.read_reachability())

    assert [link.reachable for link in links if link.kind == 'service'] == [None] * 4


@pytest.mark.asyncio
async def test_a_valkey_outage_is_unknown_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import redis

    class Down(FakeValkey):
        async def get(self, _key: str) -> str | None:
            raise redis.ConnectionError

    monkeypatch.setattr(rl, 'valkey_client', Down)
    assert await rl.read_reachability() == {}


@pytest.mark.asyncio
async def test_a_probe_failure_is_false_not_raise() -> None:
    assert await rl._probe('http://127.0.0.1:1/x') is False


def test_the_api_process_keeps_no_probe_task_or_cache() -> None:
    for name in ('_cache', 'schedule_reachability_refresh', 'refresh_reachability'):
        assert not hasattr(rl, name), name


def _routing() -> Any:
    from src.config.task_routes import ROUTES

    return ROUTES


def test_the_tick_is_routed_to_the_periodic_queue() -> None:
    from src.config.queue import TaskQueues

    assert _routing()['op.periodic.resource_reachability'] == TaskQueues.PERIODIC


@pytest.mark.asyncio
async def test_the_tick_in_gateway_mode_still_probes_enabled_services_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg(monkeypatch, gateway_subpaths=True, prometheus_port=0, dashboards_port=0)
    seen: list[str] = []

    async def fake_probe(url: str) -> bool:
        seen.append(url)
        return True

    monkeypatch.setattr(rl, '_probe', fake_probe)
    assert sorted(await rl.probe_enabled()) == ['grafana', 'mlflow']
    assert sorted(seen) == ['http://curation-mlflow:5000/health', 'http://grafana:3000/api/health']
