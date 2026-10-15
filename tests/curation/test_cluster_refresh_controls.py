"""#292: the refresh daemon is controlled by OP_CLUSTER_REFRESH_* and by the project's policy."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import httpx
import pytest
from projects.test_project_counts import _record

from curation.query_fakes import SettingsFakeOpenSearch
from scripts.curation import cluster_refresh_daemon as d
from src.services.curation.clustering.refresh_schedule import Policy, ScheduleBook
from src.services.curation.ingest_policy import IngestPolicyBody
from src.services.curation.ingest_policy_store import put_ingest_policy


def _args(monkeypatch: pytest.MonkeyPatch, *argv: str) -> Any:
    monkeypatch.setattr(sys, 'argv', ['daemon', *argv])
    return d._parse_args()


def test_defaults_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_CLUSTER_REFRESH_INTERVAL_S', '30')
    monkeypatch.setenv('OP_CLUSTER_REFRESH_GROWTH_THRESHOLD', '5000')
    monkeypatch.setenv('OP_CLUSTER_REFRESH_MAX_DEFERRAL_S', '60')
    monkeypatch.setenv('OP_CLUSTER_REFRESH_RETRY_BACKOFF_S', '90')
    monkeypatch.setenv('OP_CLUSTER_REFRESH_AUTO_LABEL', 'false')
    args = _args(monkeypatch)
    assert (args.interval_seconds, args.growth_threshold) == (30, 5000)
    assert (args.max_deferral_seconds, args.retry_backoff_seconds) == (60, 90)
    assert args.auto_label is False


def test_a_flag_overrides_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_CLUSTER_REFRESH_GROWTH_THRESHOLD', '5000')
    monkeypatch.setenv('OP_CLUSTER_REFRESH_AUTO_LABEL', 'false')
    args = _args(monkeypatch, '--growth-threshold', '10', '--auto-label')
    assert args.growth_threshold == 10
    assert args.auto_label is True


def test_without_the_environment_the_old_defaults_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        'OP_CLUSTER_REFRESH_INTERVAL_S',
        'OP_CLUSTER_REFRESH_GROWTH_THRESHOLD',
        'OP_CLUSTER_REFRESH_MAX_DEFERRAL_S',
        'OP_CLUSTER_REFRESH_RETRY_BACKOFF_S',
        'OP_CLUSTER_REFRESH_AUTO_LABEL',
    ):
        monkeypatch.delenv(name, raising=False)
    args = _args(monkeypatch)
    assert (args.interval_seconds, args.growth_threshold) == (120, 200)
    assert (args.max_deferral_seconds, args.retry_backoff_seconds) == (1800, 600)
    assert args.auto_label is True


class _Fake(SettingsFakeOpenSearch):
    async def count(self, *, body: Any = None, **_: Any) -> dict[str, int]:
        return {'count': 0 if body else 5000}


def test_a_project_with_automatic_clustering_off_is_not_refreshed() -> None:
    fake = _Fake({})
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={'status': 'idle'})

    async def go() -> ScheduleBook:
        from src.config.project_context import bind_project

        record = _record('alpha')
        with bind_project(record):
            await put_ingest_policy(
                fake,
                IngestPolicyBody(clustering={'automatic': False}),  # type: ignore[arg-type]
                expected_revision=0,
            )
        book = ScheduleBook()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await d._iteration(
                client,
                api='http://api',
                api_prefix='/curation',
                record=record,
                opensearch=fake,
                sched=book.get('alpha'),
                policy=Policy(threshold=200, max_deferral_s=1800, retry_backoff_s=600),
                auto_label=True,
                now=0.0,
            )
        return book

    book = asyncio.run(go())
    assert calls == []  # no promote, no retrain, not even a status poll
    assert book.get('alpha').job_id is None


@pytest.mark.parametrize(('automatic', 'triggered'), [(False, False), (True, True)])
def test_the_idle_retrain_follows_the_projects_clustering_switch(
    monkeypatch: pytest.MonkeyPatch, automatic: bool, triggered: bool
) -> None:
    from scripts.curation import auto_label_worker as w
    from src.config.project_context import bind_project
    from src.services.curation.clustering import retrain_policy

    async def _due(_client: Any) -> dict[str, Any]:
        return {'should': True}

    monkeypatch.setattr(retrain_policy, 'should_retrain_centroids', _due)
    fake = SettingsFakeOpenSearch({})

    async def go() -> Any:
        with bind_project(_record('alpha')):
            await put_ingest_policy(
                fake,
                IngestPolicyBody(clustering={'automatic': automatic}),  # type: ignore[arg-type]
                expected_revision=0,
            )
            return await w._maybe_auto_retrain(fake)

    assert (asyncio.run(go()) is not None) is triggered
