"""The per-photo call set every branch module uses (``frame_engine/branches/base.py``)."""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

from frame_engine.branches.base import Branch, PhotoCalls
from frame_engine.stage_metrics import StageClock
from frame_engine.transport import ModelCallError, ModelInput, Staged, TritonUnavailableError


X = ModelInput('images', 'UINT8', (1, 3, 4, 4))


class _Transport:
    """Answers every call when it is collected; records sends, collections, cancels, releases."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []
        self.held = 0
        self.refuse_send: Exception | None = None
        self.fail_collect: dict[int, Exception] = {}
        self.syncs = 0

    def stage(self, nbytes: int) -> Staged:
        self.held += nbytes
        return Staged(device=np.zeros(nbytes, np.uint8), nbytes=nbytes)

    def commit(self, staged: list[Staged], sync: Any) -> None:
        sync()
        for s in staged:
            s.committed = True

    def submit(self, model: str, staged: Staged, tensor: Any, outputs: Any, *,
               deadline: float) -> Any:  # fmt: skip
        if self.refuse_send is not None:
            raise self.refuse_send
        k = sum(1 for e in self.events if e[0] == 'send')
        self.events.append(('send', k))
        return types.SimpleNamespace(k=k, outputs=outputs)

    def collect(self, p: Any) -> dict[str, Any]:
        self.events.append(('collect', p.k))
        if p.k in self.fail_collect:
            raise self.fail_collect[p.k]
        return {p.outputs[0]: np.full((1, 2), p.k, np.float32)}

    def cancel(self, p: Any) -> None:
        self.events.append(('cancel', p.k))

    def release(self, staged: Staged, sync: Any) -> None:
        if not staged.committed:
            sync()
        self.held -= staged.nbytes


def _calls(window: int = 1 << 20) -> tuple[PhotoCalls, _Transport]:
    t = _Transport()

    def sync() -> None:
        t.syncs += 1

    return PhotoCalls(t, StageClock(), 1e18, window, sync), t  # type: ignore[arg-type]


def _send(calls: PhotoCalls, owner: Branch) -> Any:
    staged = calls.stage(X.nbytes, 'embed_crops')
    return calls.submit(owner, 'embed_pe', staged, 'pe', X, ('out',))


def test_together_sends_after_one_wait_for_the_stream() -> None:
    calls, t = _calls()
    a = b = Branch()
    with calls.together():
        first, second = _send(calls, a), _send(calls, b)
        assert t.events == []  # nothing goes out inside the round
    assert t.syncs == 1
    assert [e[0] for e in t.events] == ['send', 'send']
    assert calls.result(second)['out'][0, 0] == 1
    assert calls.result(first)['out'][0, 0] == 0
    assert t.held == 0


def test_the_window_collects_the_oldest_call_and_keeps_its_answer() -> None:
    calls, t = _calls(window=2 * X.nbytes)
    owner = Branch()
    sent = [_send(calls, owner) for _ in range(4)]
    assert t.events == [
        ('send', 0), ('send', 1), ('collect', 0), ('send', 2), ('collect', 1), ('send', 3),
    ]  # fmt: skip
    assert t.held <= 2 * X.nbytes
    assert [int(calls.result(c)['out'][0, 0]) for c in sent] == [0, 1, 2, 3]
    assert t.events.count(('collect', 0)) == 1  # the kept answer, not a second wait
    assert t.held == 0


def test_a_window_smaller_than_one_call_still_sends_it() -> None:
    calls, t = _calls(window=1)
    c = _send(calls, Branch())
    assert calls.result(c)['out'].shape == (1, 2)
    assert t.held == 0


def test_a_model_error_waits_for_its_owner_and_frees_its_slice() -> None:
    calls, t = _calls(window=X.nbytes)
    owner = Branch()
    t.fail_collect[0] = ModelCallError('pe', 'StatusCode.INTERNAL', 'boom')
    first, second = _send(calls, owner), _send(calls, owner)  # the window collects the first
    assert t.held == X.nbytes
    with pytest.raises(ModelCallError):
        calls.result(first)
    assert calls.result(second)['out'][0, 0] == 1


def test_a_call_refused_when_sent_fails_its_owner_and_gives_its_slice_back() -> None:
    calls, t = _calls()
    t.refuse_send = ModelCallError('pe', 'StatusCode.INVALID_ARGUMENT', 'bad shape')
    c = _send(calls, Branch())
    assert t.held == 0
    with pytest.raises(ModelCallError, match='bad shape'):
        calls.result(c)


def test_cancel_gives_up_only_the_failed_owners_calls() -> None:
    calls, t = _calls()
    failed, other = Branch(), Branch()
    _send(calls, failed)
    kept = _send(calls, other)
    _send(calls, failed)
    calls.cancel(failed)
    assert t.events[-2:] == [('cancel', 0), ('cancel', 2)]
    assert t.held == X.nbytes  # only the other branch's slice is still held
    assert calls.result(kept)['out'][0, 0] == 1


def test_close_cancels_what_is_in_flight_and_waits_before_it_frees_unsent_slices() -> None:
    calls, t = _calls()
    _send(calls, Branch())
    calls.stage(X.nbytes, 'embed_crops')  # kernels may still be writing it
    t.syncs = 0
    calls.close()
    assert ('cancel', 0) in t.events
    assert t.syncs == 1  # for the slice never committed, not for the sent one
    assert t.held == 0


def test_an_unreachable_server_while_making_room_fails_the_photo() -> None:
    calls, t = _calls(window=X.nbytes)
    t.fail_collect[0] = TritonUnavailableError('pe: StatusCode.UNAVAILABLE')
    _send(calls, Branch())
    with pytest.raises(TritonUnavailableError):
        _send(calls, Branch())
    calls.close()
    assert t.held == 0
