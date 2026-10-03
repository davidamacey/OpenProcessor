"""#82: the OpenSearch client logs every request (the config_revision
poll alone was ~64% of API log lines) at INFO. After configure_logging a
steady-state 200 must log nothing at INFO; failures still surface."""

from __future__ import annotations

import logging

import pytest

from src.core.logging import configure_logging


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    levels = {n: logging.getLogger(n).level for n in ('opensearch', 'elastic_transport')}
    yield
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])
    for n, lvl in levels.items():
        logging.getLogger(n).setLevel(lvl)


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.mark.parametrize('name', ['opensearch', 'elastic_transport'])
def test_steady_state_poll_logs_nothing_at_info(restore_logging, name: str) -> None:
    configure_logging(json_logs=True, log_level='INFO')
    cap = _Capture()
    logging.getLogger().addHandler(cap)
    log = logging.getLogger(name)
    log.info(
        'GET http://opensearch:9200/op_prj_x__configs/_doc/meta%3Aconfig_revision [status:200]'
    )
    assert cap.records == []
    log.warning('GET http://opensearch:9200/x [status:500]')
    assert [r.levelno for r in cap.records] == [logging.WARNING]
