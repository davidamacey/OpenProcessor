"""``request_when_free``: the live helpers honour a served ``409 <busy>`` by waiting, bounded."""

from __future__ import annotations

from typing import Any

import pytest

from tests.live import conftest as live


class _Resp:
    def __init__(self, status: int, detail: dict[str, Any] | None = None) -> None:
        self.status_code = status
        self._body = {'detail': detail} if detail is not None else {}

    def json(self) -> dict[str, Any]:
        return self._body


def _busy(error: str = 'project_busy', **extra: Any) -> _Resp:
    return _Resp(409, {'error': error, **extra})


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live.time, 'sleep', lambda _s: None)


def test_a_busy_answer_is_retried_until_the_request_goes_through() -> None:
    answers = [_busy(), _busy(), _Resp(202)]
    calls = []

    def send() -> _Resp:
        calls.append(1)
        return answers[len(calls) - 1]

    resp = live.request_when_free(send, busy='project_busy', timeout=60.0)
    assert resp.status_code == 202
    assert len(calls) == 3


def test_the_active_job_is_waited_on_when_the_busy_answer_names_it() -> None:
    answers = [_busy('job_busy', active_job_id='job-1'), _Resp(202)]
    waited: list[str] = []
    resp = live.request_when_free(
        lambda: answers.pop(0), busy='job_busy', on_active=waited.append, timeout=60.0
    )
    assert resp.status_code == 202
    assert waited == ['job-1']


def test_another_conflict_is_returned_at_once() -> None:
    other = _Resp(409, {'error': 'confirm_mismatch'})
    calls = []

    def send() -> _Resp:
        calls.append(1)
        return other

    resp = live.request_when_free(send, busy='project_busy', timeout=60.0)
    assert resp is other
    assert len(calls) == 1


def test_a_request_that_stays_busy_gives_up_at_the_deadline_with_the_last_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(live.time, 'monotonic', lambda: float(next(clock)))
    last = _busy()
    resp = live.request_when_free(lambda: last, busy='project_busy', timeout=30.0)
    assert resp is last
