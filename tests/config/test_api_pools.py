"""Each API process takes an equal share of the deployment's database connections."""

from __future__ import annotations

import pytest

from src.config.api_pools import db_pool_size
from src.config.database import DatabaseSettings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ('WEB_CONCURRENCY', 'OP_API_REPLICAS', 'OP_DB_API_CONNECTIONS_TOTAL'):
        monkeypatch.delenv(name, raising=False)


def test_the_default_install_gives_each_of_32_processes_four_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('WEB_CONCURRENCY', '32')
    assert db_pool_size() == 4


@pytest.mark.parametrize(
    ('workers', 'replicas', 'total', 'expected'),
    [(32, 2, 128, 2), (4, 1, 128, 32), (8, 1, 100, 12), (32, 4, 64, 1), (200, 1, 128, 1)],
)
def test_the_share_is_floor_of_total_over_workers_times_replicas_with_a_floor_of_one(
    monkeypatch: pytest.MonkeyPatch, workers: int, replicas: int, total: int, expected: int
) -> None:
    monkeypatch.setenv('WEB_CONCURRENCY', str(workers))
    monkeypatch.setenv('OP_API_REPLICAS', str(replicas))
    monkeypatch.setenv('OP_DB_API_CONNECTIONS_TOTAL', str(total))
    assert db_pool_size(DatabaseSettings.from_env()) == expected


@pytest.mark.parametrize('value', [None, '', '0', '-3'])
def test_a_missing_or_non_positive_worker_count_refuses_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is not None:
        monkeypatch.setenv('WEB_CONCURRENCY', value)
    with pytest.raises(ValueError, match='WEB_CONCURRENCY'):
        db_pool_size()


def test_a_malformed_worker_count_refuses_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('WEB_CONCURRENCY', 'many')
    with pytest.raises(ValueError, match='WEB_CONCURRENCY'):
        db_pool_size()
