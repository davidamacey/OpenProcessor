"""The region lifecycle against a fake model server: purge, register, re-register, unregister,
then free; and the overload and unavailable answers."""

from __future__ import annotations

import threading
import time
import types
from typing import Any

import numpy as np
import pytest
import tritonclient.grpc as grpcclient
from prometheus_client import CollectorRegistry
from tritonclient.utils import InferenceServerException

from frame_engine.slab import ALIGN
from frame_engine.stage_metrics import EngineMetrics
from frame_engine.transport import (
    BodyTransport,
    CudaShmTransport,
    ModelCallError,
    ModelInput,
    OverloadError,
    Staged,
    TritonUnavailableError,
    region_prefix,
)


CARD = '00000000:41:00.0'
PE = ModelInput('images', 'UINT8', (1, 3, 4, 4))
OUT = ('image_embeddings',)


def _infer(t: Any, staged: Staged, *, deadline: float | None = None) -> np.ndarray:
    """One encoder call, submitted and collected (what a branch does with one chunk)."""
    kw = {} if deadline is None else {'deadline': deadline}
    return t.collect(t.submit('pe', staged, PE, OUT, **kw))['image_embeddings']


def _err(code: str, msg: str) -> InferenceServerException:
    return InferenceServerException(msg=msg, status=f'StatusCode.{code}')


class FakeTriton:
    """Keeps registered regions like Triton does; ``restart`` forgets them."""

    def __init__(self, log: list[tuple[str, str]]) -> None:
        self.regions: dict[str, int] = {}
        self.up = True
        self.log = log
        self.infer_fail: list[Exception] = []
        self.unregister_fails = False
        self.lock = threading.Lock()
        self.not_found_gate: threading.Barrier | None = None

    def restart(self) -> None:
        self.regions.clear()

    def _check_up(self) -> None:
        if not self.up:
            raise _err('UNAVAILABLE', 'failed to connect to all addresses')

    def is_server_ready(self, client_timeout: float | None = None) -> bool:
        self._check_up()
        return True

    def get_cuda_shared_memory_status(self, region_name: str = '', as_json: bool = False,
                                      client_timeout: float | None = None) -> Any:  # fmt: skip
        self._check_up()
        return {'regions': {n: {'name': n, 'byte_size': str(b)} for n, b in self.regions.items()}}

    def register_cuda_shared_memory(self, name: str, raw_handle: bytes, device_id: int,
                                    byte_size: int, client_timeout: float | None = None) -> None:  # fmt: skip
        self._check_up()
        with self.lock:
            if name in self.regions:
                raise _err('ALREADY_EXISTS', f"shared memory region '{name}' already in manager")
            self.log.append(('register', name))
            self.regions[name] = byte_size

    def unregister_cuda_shared_memory(self, name: str = '',
                                      client_timeout: float | None = None) -> None:  # fmt: skip
        self._check_up()
        if self.unregister_fails:
            raise _err('INTERNAL', 'cannot unregister')
        if name in self.regions:
            self.log.append(('unregister', name))
            del self.regions[name]

    def async_infer(self, model_name: str, inputs: Any, callback: Any, *, outputs: Any = None,
                    client_timeout: Any = None) -> Any:  # fmt: skip
        """Answers through ``callback`` before it returns (tritonclient calls it on a gRPC
        thread; the transport waits for either)."""
        self._check_up()  # a refused connection fails the call before it is sent
        try:
            result, error = self._answer(model_name, inputs), None
        except InferenceServerException as exc:
            result, error = None, exc
        callback(result=result, error=error)
        return types.SimpleNamespace(cancel=lambda: None)

    def _answer(self, model_name: str, inputs: Any) -> Any:
        if self.infer_fail:
            raise self.infer_fail.pop(0)
        tensor = inputs[0]._get_tensor()
        region = tensor.parameters['shared_memory_region'].string_param if (
            'shared_memory_region' in tensor.parameters) else None  # fmt: skip
        if region is not None and region not in self.regions:
            if self.not_found_gate is not None:
                self.not_found_gate.wait(5)  # every caller has seen the loss before any recovers
            raise _err('NOT_FOUND', f"Unable to find shared memory region: '{region}'")
        self.log.append(('infer', model_name))
        return type('R', (), {'as_numpy': lambda _s, _n: np.ones((1, 4), np.float32)})()


class FakeRegion:
    def __init__(self, log: list[tuple[str, str]], name: str, nbytes: int) -> None:
        self.log, self.name = log, name
        self.raw_handle = b'aGFuZGxl'
        self.buf = np.zeros(nbytes, np.uint8)

    def view(self) -> Any:
        return self.buf

    def free(self) -> None:
        self.log.append(('free', self.name))


@pytest.fixture
def log() -> list[tuple[str, str]]:
    return []


@pytest.fixture
def server(log: list[tuple[str, str]]) -> FakeTriton:
    return FakeTriton(log)


def _cuda(server: FakeTriton, log: list[tuple[str, str]], *, nbytes: int = 64 * ALIGN,
          instance: str = 'fe0', slab_wait: float = 0.05,
          timeout: float = 5.0) -> CudaShmTransport:  # fmt: skip
    return CudaShmTransport(
        [server, server],
        EngineMetrics(CollectorRegistry()),
        timeout=timeout,
        input_factory=grpcclient.InferInput,
        output_factory=grpcclient.InferRequestedOutput,
        region_factory=lambda name, n: FakeRegion(log, name, n),
        nbytes=nbytes,
        card=CARD,
        instance=instance,
        triton_device_id=0,
        slab_wait=slab_wait,
    )


def _registrations(t: CudaShmTransport) -> float:
    return sum(
        s.value
        for m in t.metrics.registrations.collect()
        for s in m.samples
        if s.name.endswith('_total')
    )


def test_the_prefix_carries_the_card_and_the_instance() -> None:
    assert region_prefix(CARD, 'fe0') == 'op-fe-00000000-41-00-0-fe0-'
    assert region_prefix(CARD, 'fe1') != region_prefix(CARD, 'fe0')


def test_start_purges_only_this_prefix_then_registers(server: FakeTriton, log: list) -> None:
    mine = region_prefix(CARD, 'fe0')
    server.regions = {
        mine + 'deadbeef-7': 1,  # a killed predecessor on this card and slot
        mine + 'cafe-9': 1,
        region_prefix(CARD, 'fe1') + 'other-3': 1,  # another engine slot: left alone
        'someone-else': 1,
    }
    t = _cuda(server, log)
    t.start()
    assert sorted(log[:2]) == [('unregister', mine + 'cafe-9'), ('unregister', mine + 'deadbeef-7')]
    assert log[2] == ('register', t.name)
    assert t.name.startswith(mine)
    assert set(server.regions) == {t.name, region_prefix(CARD, 'fe1') + 'other-3', 'someone-else'}
    assert t.ready()


def test_a_model_server_that_starts_later_is_registered_by_the_health_check(
    server: FakeTriton, log: list
) -> None:
    server.up = False
    t = _cuda(server, log)
    t.start()
    assert not t.ready()
    assert not t.check()
    server.up = True
    assert t.check()
    assert ('register', t.name) in log


def test_an_unknown_region_is_registered_again_once_and_the_call_retried(
    server: FakeTriton, log: list
) -> None:
    t = _cuda(server, log)
    t.start()
    staged = t.stage(3 * 4 * 4)
    server.restart()  # the model server lost every registration
    out = _infer(t, staged)
    assert out.shape == (1, 4)
    assert log.count(('register', t.name)) == 2
    assert log[-1] == ('infer', 'pe')
    assert _registrations(t) == 2


def test_concurrent_callers_register_again_only_once(server: FakeTriton, log: list) -> None:
    t = _cuda(server, log)
    t.start()
    server.restart()
    staged = [t.stage(48) for _ in range(8)]
    server.not_found_gate = threading.Barrier(8)
    errors: list[BaseException] = []

    def call(s: Staged) -> None:
        try:
            _infer(t, s)
        except BaseException as exc:  # collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=call, args=(s,)) for s in staged]
    for th in threads:
        th.start()
    for th in threads:
        th.join(10)
    assert errors == []
    assert log.count(('register', t.name)) == 2  # start, then one re-registration


def test_the_retry_after_registering_again_happens_once(server: FakeTriton, log: list) -> None:
    t = _cuda(server, log)
    t.start()
    staged = t.stage(48)
    gone = _err('NOT_FOUND', f"Unable to find shared memory region: '{t.name}'")
    server.infer_fail = [gone, gone]
    with pytest.raises(ModelCallError, match='Unable to find shared memory region'):
        _infer(t, staged)
    assert log.count(('register', t.name)) == 2


def test_another_model_error_is_not_a_registration_problem(server: FakeTriton, log: list) -> None:
    t = _cuda(server, log)
    t.start()
    server.infer_fail = [_err('INVALID_ARGUMENT', 'unexpected shape for input images')]
    with pytest.raises(ModelCallError) as err:
        _infer(t, t.stage(48))
    assert err.value.code == 'StatusCode.INVALID_ARGUMENT'
    assert log.count(('register', t.name)) == 1


def test_an_unreachable_server_is_unavailable_and_the_health_check_recovers(
    server: FakeTriton, log: list
) -> None:
    t = _cuda(server, log)
    t.start()
    staged = t.stage(48)
    server.up = False
    with pytest.raises(TritonUnavailableError):
        _infer(t, staged)
    assert not t.ready()
    server.up = True
    server.restart()
    assert t.check()  # back, and the lost region registered again without traffic
    assert log.count(('register', t.name)) == 2
    _infer(t, staged)


def test_a_full_region_is_an_overload_within_the_bounded_wait(
    server: FakeTriton, log: list
) -> None:
    t = _cuda(server, log, nbytes=4 * ALIGN, slab_wait=0.05)
    t.start()
    t.stage(4 * ALIGN)
    with pytest.raises(OverloadError) as err:
        t.stage(1)
    assert err.value.cause == 'slab'


def test_release_waits_for_the_stream_before_a_slice_never_committed_is_reused(
    server: FakeTriton, log: list
) -> None:
    t = _cuda(server, log, nbytes=4 * ALIGN)
    t.start()
    s = t.stage(4 * ALIGN)
    synced: list[int] = []
    t.release(s, lambda: synced.append(t.slab.in_use))
    assert synced == [4 * ALIGN]  # waited while the slice was still held
    assert t.slab.in_use == 0


def test_one_commit_waits_for_the_stream_once_for_every_slice(
    server: FakeTriton, log: list
) -> None:
    """Round 1: every slice written, one stream wait, then all the calls go out. A committed
    slice is released without another wait (nothing writes it after the commit)."""
    t = _cuda(server, log)
    t.start()
    slices = [t.stage(48) for _ in range(3)]
    synced: list[int] = []
    t.commit(slices, lambda: synced.append(1))
    assert synced == [1]
    for s in slices:
        t.release(s, lambda: synced.append(2))
    assert synced == [1]
    assert t.slab.in_use == 0


class Delayed(FakeTriton):
    """Answers every call from another thread after ``delay`` seconds, each output named."""

    def __init__(self, log: list[tuple[str, str]], delay: float) -> None:
        super().__init__(log)
        self.delay = delay
        self.sent: list[Any] = []
        self.answered = 0

    def async_infer(self, model_name: str, inputs: Any, callback: Any, *, outputs: Any = None,
                    client_timeout: Any = None) -> Any:  # fmt: skip
        self.sent.append((model_name, inputs[0]._get_tensor(), [o.name() for o in outputs]))
        names = [o.name() for o in outputs]

        def answer() -> None:
            self.answered += 1
            arrays = {n: np.full((1, 2), k, np.float32) for k, n in enumerate(names)}
            callback(result=types.SimpleNamespace(as_numpy=arrays.__getitem__), error=None)

        threading.Timer(self.delay, answer).start()
        return types.SimpleNamespace(cancel=lambda: None)


def test_submit_returns_with_the_call_in_flight_and_collect_gives_every_named_output(
    log: list,
) -> None:
    server = Delayed(log, 0.2)
    t = _cuda(server, log)
    t.start()
    scrfd = ModelInput('input.1', 'FP32', (1, 3, 4, 4))
    outs = ('score_8', 'bbox_8', 'kps_8')
    a, b = t.stage(scrfd.nbytes), t.stage(scrfd.nbytes)
    start = time.monotonic()
    pa = t.submit('scrfd', a, scrfd, outs)
    pb = t.submit('scrfd', b, scrfd, outs)
    assert time.monotonic() - start < 0.1  # neither waited for its answer
    assert (len(server.sent), server.answered) == (2, 0)
    got = [t.collect(pa), t.collect(pb)]
    assert time.monotonic() - start < 0.35  # the two waits overlapped
    for answer in got:
        assert list(answer) == list(outs)
        assert [int(answer[n][0, 0]) for n in outs] == [0, 1, 2]


def test_the_call_carries_the_input_name_datatype_and_byte_size(log: list) -> None:
    server = Delayed(log, 0.0)
    t = _cuda(server, log)
    t.start()
    arcface = ModelInput('input', 'FP32', (2, 3, 4, 4))
    assert arcface.nbytes == 2 * 3 * 4 * 4 * 4
    s = t.stage(arcface.nbytes)
    t.collect(t.submit('arcface', s, arcface, ('output',)))
    [(model, tensor, outputs)] = server.sent
    assert (model, tensor.name, tensor.datatype, list(tensor.shape)) == (
        'arcface', 'input', 'FP32', [2, 3, 4, 4],
    )  # fmt: skip
    params = tensor.parameters
    assert params['shared_memory_region'].string_param == t.name
    assert params['shared_memory_byte_size'].int64_param == arcface.nbytes
    assert params['shared_memory_offset'].int64_param == s.offset
    assert outputs == ['output']


def test_pending_calls_never_answered_all_end_by_the_one_deadline(log: list) -> None:
    """A photo with four calls in flight against a server that went away waits about one
    timeout in total, not four: each call's end is fixed when it is submitted."""
    server = Unanswered(log)
    t = _cuda(server, log, timeout=0.3)
    t.start()
    staged = [t.stage(48) for _ in range(4)]
    pending = [t.submit('pe', s, PE, OUT) for s in staged]
    start = time.monotonic()
    for p in pending:
        with pytest.raises(TritonUnavailableError, match='no answer'):
            t.collect(p)
    assert time.monotonic() - start < 0.6
    assert server.cancelled == 4
    for s in staged:
        t.release(s, lambda: None)
    assert t.slab.in_use == 0


def test_cancel_ends_a_pending_call_without_counting_a_model_server_error(log: list) -> None:
    server = Unanswered(log)
    t = _cuda(server, log)
    t.start()
    s = t.stage(48)
    t.cancel(t.submit('pe', s, PE, OUT))
    assert server.cancelled == 1
    assert _errors(t) == {}
    assert t.ready()  # a call this engine gave up is no sign the server went away
    t.release(s, lambda: None)
    assert t.slab.in_use == 0


def test_cancel_of_an_answered_call_does_nothing(server: FakeTriton, log: list) -> None:
    t = _cuda(server, log)
    t.start()
    cancelled: list[int] = []
    real = server.async_infer

    def answered(*a: Any, **k: Any) -> Any:
        real(*a, **k)
        return types.SimpleNamespace(cancel=lambda: cancelled.append(1))

    server.async_infer = answered  # type: ignore[method-assign]
    p = t.submit('pe', t.stage(48), PE, OUT)
    t.cancel(p)
    assert cancelled == []


def test_shutdown_unregisters_before_it_frees(server: FakeTriton, log: list) -> None:
    t = _cuda(server, log)
    t.start()
    assert t.close() is True
    assert log[-2:] == [('unregister', t.name), ('free', t.name)]
    assert t.name not in server.regions
    assert not t.ready()


def test_shutdown_keeps_the_memory_when_the_server_still_imports_it(
    server: FakeTriton, log: list
) -> None:
    t = _cuda(server, log)
    t.start()
    server.unregister_fails = True
    assert t.close() is False
    assert ('free', t.name) not in log


class Unanswered(FakeTriton):
    """Takes every call and never answers it: the channel was lost with the call in flight. It
    ignores the call's deadline too, so only the transport's own bound can end the wait."""

    def __init__(self, log: list[tuple[str, str]]) -> None:
        super().__init__(log)
        self.timeouts: list[float | None] = []
        self.cancelled = 0

    def async_infer(self, model_name: str, inputs: Any, callback: Any, *, outputs: Any = None,
                    client_timeout: Any = None) -> Any:  # fmt: skip
        self.timeouts.append(client_timeout)
        return types.SimpleNamespace(cancel=self._cancel)

    def _cancel(self) -> None:
        self.cancelled += 1


class FailsMidCall(FakeTriton):
    """Takes the call, then fails it from the channel's own thread a moment later."""

    def __init__(self, log: list[tuple[str, str]], error: Exception) -> None:
        super().__init__(log)
        self.error = error

    def async_infer(self, model_name: str, inputs: Any, callback: Any, *, outputs: Any = None,
                    client_timeout: Any = None) -> Any:  # fmt: skip
        error = self.error if isinstance(self.error, InferenceServerException) else None
        if error is None:
            raise self.error  # the client refuses the call itself (closed channel)
        threading.Timer(0.05, callback, kwargs={'result': None, 'error': error}).start()
        return types.SimpleNamespace(cancel=lambda: None)


def _errors(t: Any) -> dict[tuple[str, str, str], float]:
    return {
        (s.labels['model'], s.labels['code'], s.labels['reason']): s.value
        for m in t.metrics.triton_errors.collect()
        for s in m.samples
        if s.name.endswith('_total')
    }


def test_a_call_that_is_never_answered_is_unavailable_within_its_deadline(log: list) -> None:
    server = Unanswered(log)
    t = _cuda(server, log, timeout=0.3)
    t.start()
    staged = t.stage(48)
    start = time.monotonic()
    with pytest.raises(TritonUnavailableError, match='no answer'):
        _infer(t, staged)
    assert time.monotonic() - start < 1.5
    assert server.timeouts == [0.3]
    assert server.cancelled == 1  # the call is cancelled, not left running
    assert _errors(t) == {('pe', 'StatusCode.DEADLINE_EXCEEDED', 'deadline'): 1.0}
    assert not t.ready()  # the health check decides when the server is back
    t.release(staged, lambda: None)
    assert t.slab.in_use == 0


def test_a_call_never_takes_longer_than_what_is_left_of_its_request(log: list) -> None:
    server = Unanswered(log)
    t = _cuda(server, log, timeout=30.0)
    t.start()
    staged = t.stage(48)
    start = time.monotonic()
    with pytest.raises(TritonUnavailableError):
        _infer(t, staged, deadline=time.perf_counter() + 0.2)
    assert time.monotonic() - start < 1.5
    [first] = server.timeouts
    assert first is not None
    assert first <= 0.2
    with pytest.raises(TritonUnavailableError, match='deadline'):
        _infer(t, staged, deadline=time.perf_counter() - 1)
    assert len(server.timeouts) == 1  # a request out of time makes no further call


@pytest.mark.parametrize(
    ('error', 'want'),
    [
        (_err('UNAVAILABLE', 'Socket closed'), ('StatusCode.UNAVAILABLE', 'unavailable')),
        (_err('CANCELLED', 'server shutting down'), ('StatusCode.CANCELLED', 'unavailable')),
        (ValueError('Cannot invoke RPC on closed channel!'), ('ValueError', 'client')),
    ],
)
def test_a_call_that_fails_mid_call_is_unavailable_and_counted_with_its_reason(
    log: list, error: Exception, want: tuple[str, str]
) -> None:
    server = FailsMidCall(log, error)
    t = _cuda(server, log)
    t.start()
    staged = t.stage(48)
    with pytest.raises(TritonUnavailableError):
        _infer(t, staged)
    assert _errors(t) == {('pe', *want): 1.0}
    t.release(staged, lambda: None)
    assert t.slab.in_use == 0


def test_a_failed_registration_during_a_call_is_unavailable(server: FakeTriton, log: list) -> None:
    t = _cuda(server, log)
    t.start()
    staged = t.stage(48)
    server.restart()
    real = server.register_cuda_shared_memory

    def down(*a: Any, **k: Any) -> None:
        raise _err('UNAVAILABLE', 'failed to connect to all addresses')

    server.register_cuda_shared_memory = down  # type: ignore[method-assign]
    with pytest.raises(TritonUnavailableError):
        _infer(t, staged)
    server.register_cuda_shared_memory = real  # type: ignore[method-assign]
    assert t.check()
    _infer(t, staged)


def test_the_body_transport_copies_to_the_host_and_sends_the_bytes(
    server: FakeTriton, log: list
) -> None:
    class _Dev(np.ndarray):
        def get(self) -> np.ndarray:
            return np.asarray(self).copy()

    t = BodyTransport(
        [server],
        EngineMetrics(CollectorRegistry()),
        timeout=5.0,
        input_factory=grpcclient.InferInput,
        output_factory=grpcclient.InferRequestedOutput,
        empty=lambda n: np.zeros(n, np.uint8).view(_Dev),
    )
    t.start()
    assert t.ready()
    s = t.stage(48)
    s.device[:] = 7
    with pytest.raises(RuntimeError, match='before commit'):
        t.submit('pe', s, PE, OUT)
    t.commit([s], lambda: None)
    assert _infer(t, s).shape == (1, 4)
    t.release(s, lambda: None)
    assert log == [('infer', 'pe')]


def test_the_body_transport_sends_the_bytes_as_the_input_datatype(log: list) -> None:
    class _Dev(np.ndarray):
        def get(self) -> np.ndarray:
            return np.asarray(self).copy()

    server = Delayed(log, 0.0)
    t = BodyTransport(
        [server],
        EngineMetrics(CollectorRegistry()),
        timeout=5.0,
        input_factory=grpcclient.InferInput,
        output_factory=grpcclient.InferRequestedOutput,
        empty=lambda n: np.zeros(n, np.uint8).view(_Dev),
    )
    t.start()
    x = ModelInput('x', 'FP32', (1, 3, 2, 2))
    s = t.stage(x.nbytes)
    s.device[:] = np.arange(12, dtype=np.float32).view(np.uint8)
    t.commit([s], lambda: None)
    t.collect(t.submit('rec', s, x, ('fetch_name_0',)))
    [(_model, tensor, _outs)] = server.sent
    assert (tensor.name, tensor.datatype, list(tensor.shape)) == ('x', 'FP32', [1, 3, 2, 2])
