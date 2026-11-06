"""Correlation ids: one ``request_id`` from the HTTP request into the log lines of a task.

The id comes from ``asgi-correlation-id`` (header ``X-Request-ID``). A task carries it in a message
header (``before_task_publish``), the worker restores it (``task_prerun``) together with
``job_id`` (the task id) and ``kind`` (the task name), and ``task_postrun`` clears them.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
import structlog
from asgi_correlation_id import correlation_id
from celery import Celery
from fastapi.testclient import TestClient

from src.core.logging import configure_logging, get_logger
from src.tasks import log_context


if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastapi import FastAPI


@pytest.fixture(autouse=True)
def _installed_and_isolated() -> Iterator[None]:
    log_context.install_log_context()
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    yield
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
    correlation_id.set(None)
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])


def _lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line.startswith('{')]


def _app_with_probe_task() -> Celery:
    app = Celery('probe', set_as_current=False, task_always_eager=True)

    @app.task(name='op.probe.log', shared=False)
    def probe() -> None:
        get_logger('probe').info('task_ran')

    return app


def test_a_request_id_header_reaches_the_task_log_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src.app.factory import create_app

    configure_logging(json_logs=True)
    application: FastAPI = create_app()
    celery_app = _app_with_probe_task()

    @application.get('/__probe')
    def probe_route() -> dict[str, str]:
        get_logger('route').info('handler_ran')
        celery_app.tasks['op.probe.log'].apply_async(task_id='job-7')
        return {}

    response = TestClient(application).get('/__probe', headers={'X-Request-ID': 'abc'})

    assert response.headers['X-Request-ID'] == 'abc'
    by_event = {line['event']: line for line in _lines(capsys)}
    assert by_event['handler_ran']['request_id'] == 'abc'
    assert by_event['task_ran']['request_id'] == 'abc'
    assert by_event['task_ran']['job_id'] == 'job-7'
    assert by_event['task_ran']['kind'] == 'op.probe.log'


def test_a_request_without_the_header_gets_a_generated_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src.app.factory import create_app

    configure_logging(json_logs=True)
    application: FastAPI = create_app()

    @application.get('/__probe')
    def probe_route() -> dict[str, str]:
        get_logger('route').info('handler_ran')
        return {}

    response = TestClient(application).get('/__probe')

    generated = response.headers['X-Request-ID']
    assert generated
    assert {line['event']: line for line in _lines(capsys)}['handler_ran'][
        'request_id'
    ] == generated


def test_an_unsafe_request_id_is_replaced() -> None:
    from src.app.factory import create_app

    response = TestClient(create_app()).get('/live', headers={'X-Request-ID': 'a b\tc'})
    assert response.headers['X-Request-ID'] not in {'a b\tc', ''}


def test_the_id_is_copied_into_the_message_headers_on_publish() -> None:
    headers: dict[str, str] = {}
    correlation_id.set('req-1')
    log_context.transfer_request_id(headers=headers)
    assert headers == {log_context.HEADER: 'req-1'}


def test_nothing_is_added_to_the_headers_without_a_request() -> None:
    headers: dict[str, str] = {}
    correlation_id.set(None)
    log_context.transfer_request_id(headers=headers)
    assert headers == {}


class _Request:
    def __init__(self, header: str | None) -> None:
        self._header = header

    def get(self, key: str) -> str | None:
        return self._header if key == log_context.HEADER else None


class _Task:
    name = 'op.probe.log'

    def __init__(self, header: str | None) -> None:
        self.request = _Request(header)


def test_the_worker_restores_the_id_and_binds_job_id_and_kind(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(json_logs=True)
    correlation_id.set(None)
    log_context.restore_request_id(task_id='job-9', task=_Task('req-2'))
    get_logger('w').info('in_task')
    log_context.clear_request_id()
    get_logger('w').info('after_task')

    inside, after = _lines(capsys)
    assert (inside['request_id'], inside['job_id'], inside['kind']) == (
        'req-2',
        'job-9',
        'op.probe.log',
    )
    assert 'job_id' not in after
    assert after.get('request_id') in (None, '-')


def test_a_task_with_no_request_gets_its_own_id() -> None:
    correlation_id.set(None)
    log_context.restore_request_id(task_id='job-1', task=_Task(None))
    first = correlation_id.get()
    log_context.clear_request_id()
    log_context.restore_request_id(task_id='job-2', task=_Task(None))
    second = correlation_id.get()
    assert first
    assert second
    assert first != second


def test_a_real_publish_carries_the_header_and_the_worker_side_restores_it() -> None:
    from src.clients import task_queue

    app = task_queue._build_producer('memory://')
    correlation_id.set('req-3')
    app.send_task('op.infra.noop', args=[str(uuid4())], task_id='job-3')
    with app.connection_for_read() as connection, connection.SimpleQueue('jobs') as queue:
        message = queue.get(block=False)
        carried = message.headers.get(log_context.HEADER)
        message.ack()
    assert carried == 'req-3'
    correlation_id.set(None)
    log_context.restore_request_id(task_id='job-3', task=_Task(carried))
    assert correlation_id.get() == 'req-3'
