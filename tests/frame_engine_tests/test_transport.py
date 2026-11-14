"""The region lifecycle against a fake model server: purge, register, re-register, unregister,
then free; and the overload and unavailable answers."""

from __future__ import annotations

import threading
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
    OverloadError,
    Staged,
    TritonUnavailableError,
    region_prefix,
)


CARD = '00000000:41:00.0'


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

    def infer(self, model_name: str, inputs: Any, *, outputs: Any = None,
              client_timeout: Any = None) -> Any:  # fmt: skip
        self._check_up()
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
          instance: str = 'fe0', slab_wait: float = 0.05) -> CudaShmTransport:  # fmt: skip
    return CudaShmTransport(
        [server, server],
        EngineMetrics(CollectorRegistry()),
        timeout=5.0,
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
    out = t.infer('pe', staged, 0, [1, 3, 4, 4], 'image_embeddings')
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
            t.infer('pe', s, 0, [1, 3, 4, 4], 'image_embeddings')
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
        t.infer('pe', staged, 0, [1, 3, 4, 4], 'image_embeddings')
    assert log.count(('register', t.name)) == 2


def test_another_model_error_is_not_a_registration_problem(server: FakeTriton, log: list) -> None:
    t = _cuda(server, log)
    t.start()
    server.infer_fail = [_err('INVALID_ARGUMENT', 'unexpected shape for input images')]
    with pytest.raises(ModelCallError) as err:
        t.infer('pe', t.stage(48), 0, [1, 3, 4, 4], 'image_embeddings')
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
        t.infer('pe', staged, 0, [1, 3, 4, 4], 'image_embeddings')
    assert not t.ready()
    server.up = True
    server.restart()
    assert t.check()  # back, and the lost region registered again without traffic
    assert log.count(('register', t.name)) == 2
    t.infer('pe', staged, 0, [1, 3, 4, 4], 'image_embeddings')


def test_a_full_region_is_an_overload_within_the_bounded_wait(
    server: FakeTriton, log: list
) -> None:
    t = _cuda(server, log, nbytes=4 * ALIGN, slab_wait=0.05)
    t.start()
    t.stage(4 * ALIGN)
    with pytest.raises(OverloadError) as err:
        t.stage(1)
    assert err.value.cause == 'slab'


def test_release_waits_for_the_stream_before_the_slice_is_reused(
    server: FakeTriton, log: list
) -> None:
    t = _cuda(server, log, nbytes=4 * ALIGN)
    t.start()
    s = t.stage(4 * ALIGN)
    synced: list[int] = []
    t.release(s, lambda: synced.append(t.slab.in_use))
    assert synced == [4 * ALIGN]  # waited while the slice was still held
    assert t.slab.in_use == 0


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
    s = t.stage(2 * 48)
    s.device[48:] = 7
    with pytest.raises(RuntimeError, match='before commit'):
        t.infer('pe', s, 48, [1, 3, 4, 4], 'image_embeddings')
    t.commit(s)
    assert t.infer('pe', s, 48, [1, 3, 4, 4], 'image_embeddings').shape == (1, 4)
    t.release(s, lambda: None)
    assert log == [('infer', 'pe')]
