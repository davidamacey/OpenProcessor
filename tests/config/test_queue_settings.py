"""``src/config/queue.py``: the queue names, the broker settings, the transport options."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.config.env import EnvError
from src.config.job_kinds import KINDS
from src.config.queue import (
    QUEUE_PROFILE,
    QueueSettings,
    TaskQueues,
    broker_transport_options,
    declared_queues,
)
from src.config.task_routes import ROUTES


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def password_file(tmp_path: Path) -> Path:
    path = tmp_path / 'password'
    path.write_text('s3cret\n')
    return path


def _env(monkeypatch: pytest.MonkeyPatch, url: str, file: Path | str) -> None:
    monkeypatch.setenv('OP_QUEUE_URL', url)
    monkeypatch.setenv('OP_QUEUE_PASSWORD_FILE', str(file))


def test_valid_settings_load_and_the_password_is_not_in_the_url(
    monkeypatch: pytest.MonkeyPatch, password_file: Path
) -> None:
    _env(monkeypatch, 'redis://valkey:6379/0', password_file)
    settings = QueueSettings.from_env()
    assert settings.url == 'redis://valkey:6379/0'
    assert settings.password == 's3cret'
    assert settings.broker_url == 'redis://:s3cret@valkey:6379/0'
    assert 's3cret' not in repr(settings)


def test_the_default_url_is_the_compose_service(
    monkeypatch: pytest.MonkeyPatch, password_file: Path
) -> None:
    monkeypatch.delenv('OP_QUEUE_URL', raising=False)
    monkeypatch.setenv('OP_QUEUE_PASSWORD_FILE', str(password_file))
    assert QueueSettings.from_env().url == 'redis://valkey:6379/0'


def test_rediss_is_accepted(monkeypatch: pytest.MonkeyPatch, password_file: Path) -> None:
    _env(monkeypatch, 'rediss://valkey:6380/1', password_file)
    assert QueueSettings.from_env().broker_url.startswith('rediss://:s3cret@')


@pytest.mark.parametrize(
    'url',
    [
        'valkey://valkey:6379/0',
        'amqp://valkey/0',
        'redis://user:pw@valkey:6379/0',
        'redis://:pw@valkey/0',
    ],
)
def test_a_bad_url_names_the_variable(
    monkeypatch: pytest.MonkeyPatch, password_file: Path, url: str
) -> None:
    _env(monkeypatch, url, password_file)
    with pytest.raises(EnvError, match='OP_QUEUE_URL'):
        QueueSettings.from_env()


def test_a_missing_password_file_names_the_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _env(monkeypatch, 'redis://valkey:6379/0', tmp_path / 'absent')
    with pytest.raises(EnvError, match='OP_QUEUE_PASSWORD_FILE'):
        QueueSettings.from_env()


def test_an_unset_or_empty_password_file_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv('OP_QUEUE_PASSWORD_FILE', raising=False)
    with pytest.raises(EnvError, match='OP_QUEUE_PASSWORD_FILE'):
        QueueSettings.from_env()
    empty = tmp_path / 'empty'
    empty.write_text('\n')
    _env(monkeypatch, 'redis://valkey:6379/0', empty)
    with pytest.raises(EnvError, match='OP_QUEUE_PASSWORD_FILE'):
        QueueSettings.from_env()


def test_all_equals_the_routed_queues_plus_the_declared_ones() -> None:
    routed = set(ROUTES.values())
    declared = set(QUEUE_PROFILE)
    assert set(TaskQueues.ALL) == routed | declared
    assert len(TaskQueues.ALL) == len(set(TaskQueues.ALL))
    assert routed <= set(TaskQueues.ALL)


def test_every_kind_routes_to_its_own_queue() -> None:
    for kind in KINDS.values():
        assert ROUTES[kind.task_name] == kind.queue, kind.name


def test_transport_options_are_a_fresh_equal_dict_each_call() -> None:
    first, second = broker_transport_options(), broker_transport_options()
    assert first == second
    assert first is not second
    assert first['visibility_timeout'] == 21600
    assert first['retry_on_timeout'] is False


def test_each_queue_has_its_own_exchange_and_routing_key() -> None:
    """A bare ``Queue('jobs')`` binds to exchange ``celery`` / key ``celery``: every queue would
    receive a copy of every message (found by spike S1 on the Redis transport)."""
    queues = declared_queues()
    assert [queue.name for queue in queues] == list(TaskQueues.ALL)
    for queue in queues:
        assert queue.exchange.name == queue.name
        assert queue.routing_key == queue.name
