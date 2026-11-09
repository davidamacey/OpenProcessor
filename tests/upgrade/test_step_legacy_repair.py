"""2E-3: the legacy-repair step (moved from the runtime module): the embedded-state backfill is
targeted, throttled, single-sliced, retried while OpenSearch answers 429, and never started when
nothing is legacy; the region class is added only when no active class carries its name."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from typing import Any

import pytest

from src.domain.embedding_state import legacy_embedded_clause
from src.services.upgrade.from_060.admin_client import ControlWriter
from src.services.upgrade.from_060.model import (
    ActivationSrc,
    ClassSrc,
    ConfigDocSrc,
    ConfigRevisionSrc,
    ProjectData,
    RegistrySrc,
)
from src.services.upgrade.from_060.steps import legacy_repair


T = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


class _BusyError(Exception):
    status_code = 429


class _Client:
    """The raw OpenSearch client calls the writer issues."""

    def __init__(self, legacy: int = 7, failures: int = 0) -> None:
        self.legacy, self.failures = legacy, failures
        self.calls: list[dict[str, Any]] = []
        self.tasks = self
        self.polled: list[str] = []

    async def get(self, task_id: str) -> dict[str, Any]:
        self.polled.append(task_id)
        return {'completed': True, 'response': {'updated': 7}}

    async def count(self, **_kw: Any) -> dict[str, Any]:
        return {'count': self.legacy}

    async def update_by_query(self, **kw: Any) -> dict[str, Any]:
        if self.failures:
            self.failures -= 1
            raise _BusyError('rejected_execution_exception')
        self.calls.append(kw)
        return {'task': 'node:1'}


def _ctx(client: _Client) -> Any:
    return SimpleNamespace(writer=ControlWriter(client))


@pytest.fixture(autouse=True)
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []

    async def _sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(legacy_repair.asyncio, 'sleep', _sleep)
    return waits


@pytest.mark.asyncio
async def test_the_backfill_targets_only_vectored_items_without_a_state() -> None:
    client = _Client()
    assert await legacy_repair.backfill_embedded_state(_ctx(client), 'items') == 7
    [call] = client.calls
    assert call['index'] == 'items'
    assert call['body']['query'] == legacy_embedded_clause('pe_embedding', 'embedding_state')
    assert "'embedded'" in call['body']['script']['source']
    assert call['conflicts'] == 'proceed'
    assert call['wait_for_completion'] is False
    assert client.polled == ['node:1']
    assert call['body']['query']['bool']['must_not'] == [{'exists': {'field': 'embedding_state'}}]


@pytest.mark.asyncio
async def test_the_backfill_is_throttled_and_single_sliced() -> None:
    client = _Client()
    await legacy_repair.backfill_embedded_state(_ctx(client), 'items')
    [call] = client.calls
    assert call['slices'] == 1
    assert 0 < call['requests_per_second'] <= 1000


@pytest.mark.asyncio
async def test_nothing_legacy_starts_no_task() -> None:
    client = _Client(legacy=0)
    assert await legacy_repair.backfill_embedded_state(_ctx(client), 'items') == 0
    assert client.calls == []


@pytest.mark.asyncio
async def test_a_429_is_retried_with_backoff_not_dropped(slept: list[float]) -> None:
    client = _Client(failures=2)
    assert await legacy_repair.backfill_embedded_state(_ctx(client), 'items') == 7
    assert slept == [2.0, 4.0]


@pytest.mark.asyncio
async def test_a_persistent_429_raises_so_the_upgrade_stops() -> None:
    with pytest.raises(_BusyError):
        await legacy_repair.backfill_embedded_state(_ctx(_Client(failures=99)), 'items')


@pytest.mark.asyncio
async def test_a_non_429_error_is_not_retried(slept: list[float]) -> None:
    class _Broken(_Client):
        async def update_by_query(self, **_kw: Any) -> dict[str, Any]:
            raise ValueError('bad request')

    with pytest.raises(ValueError, match='bad request'):
        await legacy_repair.backfill_embedded_state(_ctx(_Broken()), 'items')
    assert slept == []


def _project(classes: list[tuple[int, str, bool]], region: str | None) -> ProjectData:
    revision = ConfigRevisionSrc(1, {'region_class_name': region} if region else {}, T, None)
    doc = ConfigDocSrc(None, 'region_profile', 'cars', '', (revision,), T, T)
    activation = ActivationSrc('detection_profile', None, None, 'cars', 1, 'region_profile', T)
    registry = RegistrySrc(
        tuple(ClassSrc(i, n, None, dep, None, None, '', T) for i, n, dep in classes),
        10,
        0,
        'class_registry_file',
        T,
    )
    return ProjectData(
        project=SimpleNamespace(slug='p'),  # type: ignore[arg-type]
        registry=registry,
        configs=(doc,),
        activations=(activation,),
    )


def test_the_region_class_is_added_when_no_active_class_carries_its_name() -> None:
    assert legacy_repair.region_class_name(_project([], 'plate')) == 'plate'
    assert legacy_repair.adds_region_class(_project([(0, 'car', False)], 'plate'))


def test_it_is_not_added_when_an_active_class_has_the_name_by_the_one_rule() -> None:
    assert not legacy_repair.adds_region_class(_project([(0, 'Road Sign', False)], 'road_sign'))


def test_a_deprecated_class_does_not_count_and_no_profile_class_adds_nothing() -> None:
    assert legacy_repair.adds_region_class(_project([(0, 'plate', True)], 'plate'))
    assert not legacy_repair.adds_region_class(_project([(0, 'car', False)], None))
