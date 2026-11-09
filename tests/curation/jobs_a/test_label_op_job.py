"""label_op and label_undo on the queue: one row and one message, busy, the run (4A-5).

The relabel and undo behaviours themselves are covered through the routes in
``tests/curation/test_label_op_record.py`` and ``test_label_op_undo.py``."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - pytest fixture annotation

import pytest
from fixtures.job_rows import sql

from curation.label_ops_env import build_env, proposal


pytest_plugins = ['_records_world', 'curation.jobs_a.autoqueue']
pytestmark = [pytest.mark.db, pytest.mark.usefixtures('world')]


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    return build_env(monkeypatch, tmp_path, {f'c{i}': proposal(f'c{i}') for i in range(4)})


def test_a_relabel_queues_one_row_and_one_message_and_records_the_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pg, producer
) -> None:
    env = _env(monkeypatch, tmp_path)
    resp = env.queue_label('widget', crop_ids=['c0', 'c1'])
    assert resp.status_code == 202
    body = resp.json()
    assert body['status'] == 'queued'
    assert producer.sent() == [body['job_id']]
    assert sql(pg, 'SELECT count(*) FROM job_items WHERE job_id = %s', body['job_id']) == [(2,)]
    assert env.record(body['op_id'])['status'] == 'writing'
    assert 'label_op_id' not in env.docs()['c0'], 'nothing is written by the request'


def test_the_run_writes_and_records_the_operation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    done = env.label('widget', crop_ids=['c0', 'c1', 'c2'])
    assert sorted(done['updated_ids']) == ['c0', 'c1', 'c2']
    assert env.record(done['op_id'])['status'] == 'written'


def test_a_second_relabel_of_the_same_operation_id_scope_is_independent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    first = env.queue_label('widget', crop_ids=['c0']).json()
    second = env.queue_label('gadget', crop_ids=['c1']).json()
    assert first['op_id'] != second['op_id']  # exclusivity is per operation id
