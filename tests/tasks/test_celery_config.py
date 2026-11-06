"""The Celery configuration of the job plan (02b section 7.1), for all three app factories."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from celery import Celery

from src.config.queue import TaskQueues
from src.tasks.runtime import gpu_app


@pytest.fixture
def worker_conf(queue_env: None) -> Any:
    from src.tasks.app import app

    return app.conf


def test_limits_are_ordered_visibility_over_hard_over_soft(worker_conf: Any) -> None:
    visibility = worker_conf.broker_transport_options['visibility_timeout']
    assert visibility > worker_conf.task_time_limit > worker_conf.task_soft_time_limit


def test_delivery_and_serialization_settings(worker_conf: Any) -> None:
    conf = worker_conf
    assert conf.worker_prefetch_multiplier == 1
    assert conf.task_acks_late is True
    assert conf.task_retry_on_worker_lost is True
    assert conf.task_serializer == 'json'
    assert conf.result_serializer == 'json'
    assert list(conf.accept_content) == ['json']
    assert conf.task_ignore_result is True
    assert conf.task_track_started is False


def test_no_result_backend(worker_conf: Any) -> None:
    assert not worker_conf.result_backend


def test_shutdown_and_connection_loss_settings(worker_conf: Any) -> None:
    conf = worker_conf
    assert conf.worker_soft_shutdown_timeout == 30
    assert conf.worker_enable_soft_shutdown_on_idle is True
    assert conf.worker_cancel_long_running_tasks_on_connection_loss is False
    assert conf.broker_connection_max_retries is None
    assert conf.broker_connection_retry_on_startup is True
    assert conf.worker_send_task_events is True


def test_no_missing_queues_and_every_declared_queue_exists(worker_conf: Any) -> None:
    assert worker_conf.task_create_missing_queues is False
    assert {queue.name for queue in worker_conf.task_queues} == set(TaskQueues.ALL)


def test_the_broker_url_carries_the_password_from_the_file(worker_conf: Any) -> None:
    assert worker_conf.broker_url == 'redis://:test-password@valkey:6379/0'


def test_all_task_modules_are_included(worker_conf: Any) -> None:
    from src.tasks.app import TASK_MODULES

    assert tuple(worker_conf.include) == TASK_MODULES


def test_the_worker_app_and_the_gpu_app_carry_equal_broker_transport_options(
    queue_env: None, worker_conf: Any
) -> None:
    gpu = gpu_app.make_app([])
    assert worker_conf.broker_transport_options == gpu.conf.broker_transport_options


def test_worker_ready_refuses_an_unrouted_task(queue_env: None) -> None:
    app = Celery('probe', set_as_current=False)

    @app.task(name='op.probe.unrouted', shared=False)
    def unrouted() -> None: ...

    with pytest.raises(SystemExit, match=r'op\.probe\.unrouted'):
        gpu_app.validate_task_routes(app)


def test_worker_ready_refuses_a_routed_task_without_a_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(gpu_app.ROUTES, 'op.probe.kindless', TaskQueues.JOBS)
    app = Celery('probe', set_as_current=False)

    @app.task(name='op.probe.kindless', shared=False)
    def kindless() -> None: ...

    with pytest.raises(SystemExit, match=r'without a job kind: \[.op.probe.kindless.\]'):
        gpu_app.validate_task_routes(app)


def test_a_routed_task_with_a_kind_passes() -> None:
    app = Celery('probe', set_as_current=False)

    @app.task(name='op.infra.noop', shared=False)
    def noop() -> None: ...

    gpu_app.validate_task_routes(app)


def test_process_init_disposes_inherited_engines_when_the_module_was_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fake = types.ModuleType('src.db.engine')
    fake.dispose_inherited_engines = lambda: calls.append('disposed')  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'src.db.engine', fake)
    gpu_app._on_worker_process_init()
    assert calls == ['disposed']


def test_process_init_does_nothing_when_no_engine_module_was_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, 'src.db.engine', raising=False)
    gpu_app._on_worker_process_init()
    assert 'src.db.engine' not in sys.modules


def test_worker_init_rejects_retired_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.tasks.app import _on_worker_init

    monkeypatch.setenv('VLM_URL', 'http://x')
    with pytest.raises(RuntimeError, match='VLM_URL'):
        _on_worker_init()
