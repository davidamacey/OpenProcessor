"""How a model input reaches Triton: one CUDA shared-memory region, or the plain request body.

The engine calls the TensorRT models as a Triton gRPC client. A photo's model inputs are written
by the kernels into a device buffer (:meth:`Transport.stage`), made visible to the model server
(:meth:`Transport.commit`, one stream wait for any number of slices), sent without waiting for the
answer (:meth:`Transport.submit` returns a :class:`Pending`), collected by the request deadline
(:meth:`Transport.collect`, every named output of the call) or given up (:meth:`Transport.cancel`),
and given back (:meth:`Transport.release`). Two implementations, chosen by
``OP_FRAME_ENGINE_TRANSPORT``:

* :class:`CudaShmTransport` (``cuda``): one device allocation per process, exported as a CUDA IPC
  handle and registered with Triton once (KServe shared-memory extension, Triton 26.10
  ``CudaSharedMemoryRegister``); every input is a slice of it, addressed by offset
  (:mod:`frame_engine.slab`), so nothing is copied to the host.
* :class:`BodyTransport` (``body``): the input is copied to the host and sent in the request.

Region rules (all mandatory, see the plan's D1):

* The name is a fixed prefix, the card and the engine instance (:func:`region_prefix`), then the
  process identity. At start every registered region with this card and instance's prefix is
  unregistered first: a killed predecessor leaves its region registered and its memory pinned in
  the model server until that happens.
* A call answered "Unable to find shared memory region" (the model server restarted and lost its
  registrations) re-registers once under a single-flight lock and is sent again once, from the
  same slice, when it is collected. The health check (:meth:`CudaShmTransport.check`) re-registers
  too when the server is back and the region is missing, so readiness returns without traffic.
* Shutdown unregisters first, then frees: the exporting process must outlive every import (CUDA
  documents freeing exported memory before the importer closes it as undefined behaviour). The
  memory is freed only after a successful unregister; otherwise the process exit releases it.
* The slab wait is bounded; a timeout is :class:`OverloadError`.
* A slice goes back to the slab only after its call ended (answered, failed, or cancelled) and
  after the worker stream was waited for since the last kernel that wrote it: the commit's wait
  counts, so a committed slice is released without another one.

Every model call has a deadline: ``OP_FRAME_ENGINE_TRITON_TIMEOUT_SECONDS``, or what is left of the
request's deadline when that is sooner, fixed when the call is submitted. The call is issued with
tritonclient's ``async_infer`` (tritonclient 2.74.0 ``InferenceServerClient.async_infer``: the
callback gets ``(result, error)`` on a gRPC thread, ``client_timeout`` is the gRPC deadline, and the
returned ``CallContext.cancel`` cancels the underlying gRPC future), and :meth:`collect` waits for the
answer no longer than that deadline, then cancels the call: a channel lost with calls in flight
cannot hold a worker, its slices or its budget, and a photo with several calls in flight waits one
deadline, not one per call. A call without an answer in time, a channel or server that went away
(UNAVAILABLE, CANCELLED) and a call the client library itself fails are
:class:`TritonUnavailableError` (the request is answered UNAVAILABLE, retryable); an error the
server answered is :class:`ModelCallError` (the branch fails). Each failure is counted with its
reason in ``op_frame_engine_triton_call_errors_total``; a call the engine cancels itself (a sibling
failed) is not a model-server failure and is not counted.
"""

from __future__ import annotations

import base64
import itertools
import logging
import math
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from tritonclient.utils import InferenceServerException, triton_to_np_dtype

from frame_engine.slab import SlabAllocator, SlabTimeoutError


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from frame_engine.stage_metrics import EngineMetrics


logger = logging.getLogger('frame_engine.transport')

REGION_PREFIX = 'op-fe-'
_STATUS_DEADLINE = 'StatusCode.DEADLINE_EXCEEDED'
#: Codes that mean the server or the channel is gone (retryable), with the metric reason.
_GONE = {
    'StatusCode.UNAVAILABLE': 'unavailable',
    'StatusCode.CANCELLED': 'unavailable',
    _STATUS_DEADLINE: 'deadline',
}
_REGION_UNKNOWN = 'Unable to find shared memory region'


class OverloadError(Exception):
    """The engine is full right now (slab, budget or admission): a retryable answer."""

    def __init__(self, cause: str, detail: str = '') -> None:
        self.cause = cause
        super().__init__(f'{cause}: {detail}' if detail else cause)


class TritonUnavailableError(Exception):
    """The model server did not answer (down, restarting, deadline): retryable."""


class ModelCallError(Exception):
    """The model server answered a call with an error (the branch fails, the engine goes on)."""

    def __init__(self, model: str, code: str, message: str) -> None:
        self.model = model
        self.code = code
        super().__init__(f'{model}: {code} {message}')


class DeviceRegion(Protocol):
    """One exported device allocation."""

    raw_handle: bytes  # the serialized cudaIpcMemHandle_t, base64 (what the register call takes)

    def view(self) -> Any: ...
    def free(self) -> None: ...


def region_prefix(card: str, instance: str) -> str:
    """The prefix every region of this card's engine ``instance`` carries (purged at start)."""
    safe = ''.join(c if c.isalnum() else '-' for c in card.lower())
    return f'{REGION_PREFIX}{safe}-{instance}-'


def process_identity() -> str:
    """Host name (the container id under Docker) and process id: unique per live process."""
    return f'{socket.gethostname()}-{os.getpid()}'


@dataclass(eq=False)
class Staged:
    """One staged input: a flat uint8 device array and how it reaches the server (a slice is
    itself: compared by identity, never by its bytes)."""

    device: Any
    nbytes: int
    offset: int = -1  # slice of the region (cuda); -1 for the body transport
    host: np.ndarray | None = None  # the body transport's host copy, after commit
    #: The stream was waited for after the kernels wrote it; nothing writes it afterwards.
    committed: bool = False


@dataclass(frozen=True)
class ModelInput:
    """The one input tensor of a model call: its name, KServe datatype and shape (batch first)."""

    name: str
    datatype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * int(np.dtype(triton_to_np_dtype(self.datatype)).itemsize)


def _code(exc: BaseException) -> str:
    status = getattr(exc, 'status', None)
    return str(status() if callable(status) else status or type(exc).__name__)


def _message(exc: BaseException) -> str:
    message = getattr(exc, 'message', None)
    return str(message() if callable(message) else exc)


@dataclass
class _Answer:
    """Where ``async_infer``'s callback (run on a gRPC thread) leaves the result or the error."""

    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: InferenceServerException | None = None

    def __call__(self, result: Any, error: InferenceServerException | None) -> None:
        self.result, self.error = result, error
        self.done.set()


@dataclass(eq=False)
class Pending:
    """One model call in flight: :meth:`_Base.collect` it or :meth:`_Base.cancel` it."""

    model: str
    staged: Staged
    tensor: ModelInput
    outputs: tuple[str, ...]
    deadline: float  # the request's (``time.perf_counter``)
    generation: int  # the region registration the call was sent under
    timeout: float = 0.0
    ends: float = 0.0  # ``time.perf_counter`` value after which :meth:`_Base.collect` gives up
    answer: _Answer = field(default_factory=_Answer)
    call: Any = None  # tritonclient's ``CallContext``


class _Base:
    """The model-server channels and the infer call both transports share."""

    def __init__(
        self,
        clients: Sequence[Any],
        metrics: EngineMetrics,
        *,
        timeout: float,
        input_factory: Callable[..., Any],
        output_factory: Callable[[str], Any],
    ) -> None:
        if not clients:
            raise ValueError('a transport needs at least one model-server channel')
        self.clients = list(clients)
        self.metrics = metrics
        self.timeout = timeout
        self._input = input_factory
        self._output = output_factory
        self._rr = itertools.count()
        self.server_ready = False
        self.generation = 0  # bumped by every region registration (the cuda transport)

    def _client(self) -> Any:
        return self.clients[next(self._rr) % len(self.clients)]

    def _unavailable(self, model: str, code: str, reason: str, text: str) -> TritonUnavailableError:
        self.metrics.triton_errors.labels(model, code, reason).inc()
        self.server_ready = False  # the health check says when the server answers again
        return TritonUnavailableError(f'{model}: {code} {text}')

    def _failed(self, model: str, exc: InferenceServerException) -> Exception:
        code = _code(exc)
        if code in _GONE:
            return self._unavailable(model, code, _GONE[code], _message(exc))
        self.metrics.triton_errors.labels(model, code, 'rejected').inc()
        return ModelCallError(model, code, _message(exc))

    def submit(
        self,
        model: str,
        staged: Staged,
        tensor: ModelInput,
        outputs: Sequence[str],
        *,
        deadline: float = math.inf,
    ) -> Pending:
        """Send one call reading ``staged`` (committed) and return without waiting for it."""
        pending = Pending(model, staged, tensor, tuple(outputs), deadline, self.generation)
        self._send(pending)
        return pending

    def _send(self, p: Pending) -> None:
        inp = self._tensor(p.staged, p.tensor)
        timeout = min(self.timeout, p.deadline - time.perf_counter())
        if timeout <= 0:
            raise self._unavailable(
                p.model, _STATUS_DEADLINE, 'deadline', 'request deadline passed'
            )
        p.answer = _Answer()
        try:
            p.call = self._client().async_infer(
                p.model,
                [inp],
                p.answer,
                outputs=[self._output(name) for name in p.outputs],
                client_timeout=timeout,
            )
        except InferenceServerException as exc:
            raise self._failed(p.model, exc) from exc
        except Exception as exc:  # the client refused the call itself (a closed channel)
            logger.warning('%s: the client failed the call', p.model, exc_info=True)
            raise self._unavailable(p.model, type(exc).__name__, 'client', str(exc)) from exc
        p.timeout = timeout
        p.ends = time.perf_counter() + timeout

    def _wait(self, p: Pending) -> dict[str, np.ndarray]:
        """The call's outputs, or its error; it ends by its deadline, answered or not."""
        if not p.answer.done.wait(max(0.0, p.ends - time.perf_counter())):
            p.call.cancel()
            raise self._unavailable(
                p.model, _STATUS_DEADLINE, 'deadline', f'no answer within {p.timeout:.2f} s'
            )
        if p.answer.error is not None:
            raise self._failed(p.model, p.answer.error) from p.answer.error
        return {name: p.answer.result.as_numpy(name) for name in p.outputs}

    def collect(self, p: Pending) -> dict[str, np.ndarray]:
        return self._wait(p)

    @staticmethod
    def cancel(p: Pending) -> None:
        """Give up a call this engine no longer needs (a sibling failed); its slice may then be
        released (the module docstring's rule)."""
        if p.call is not None and not p.answer.done.is_set():
            p.call.cancel()

    def _tensor(self, staged: Staged, tensor: ModelInput) -> Any:
        raise NotImplementedError

    def check_server(self) -> bool:
        try:
            self.server_ready = bool(self.clients[0].is_server_ready(client_timeout=self.timeout))
        except InferenceServerException as exc:
            logger.warning('model server not ready: %s', exc)
            self.server_ready = False
        return self.server_ready


class BodyTransport(_Base):
    """Inputs travel in the request body: a device-to-host copy per input, no coupling."""

    def __init__(self, *args: Any, empty: Callable[[int], Any], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._empty = empty

    def start(self) -> None:
        self.check_server()

    def stage(self, nbytes: int) -> Staged:
        return Staged(device=self._empty(nbytes), nbytes=nbytes)

    @staticmethod
    def commit(staged: Sequence[Staged], _sync: Callable[[], None]) -> None:
        for s in staged:  # a blocking copy on the worker stream: it waits for the kernels
            s.host = s.device.get()
            s.committed = True

    def _tensor(self, staged: Staged, tensor: ModelInput) -> Any:
        if staged.host is None:
            raise RuntimeError('submit before commit')
        dtype = np.dtype(triton_to_np_dtype(tensor.datatype))
        inp = self._input(tensor.name, list(tensor.shape), tensor.datatype)
        data = staged.host[: tensor.nbytes].view(dtype).reshape(tensor.shape)
        inp.set_data_from_numpy(data)
        return inp

    @staticmethod
    def release(staged: Staged, sync: Callable[[], None]) -> None:
        if not staged.committed:
            sync()  # no kernel may still write the block when the pool hands it out again
        staged.device = None
        staged.host = None

    def ready(self) -> bool:
        return self.server_ready

    def check(self) -> bool:
        return self.check_server()

    def close(self) -> None:
        return None


class CudaShmTransport(_Base):
    """One registered CUDA shared-memory region per process; inputs are slices of it."""

    def __init__(
        self,
        *args: Any,
        region_factory: Callable[[str, int], DeviceRegion],
        nbytes: int,
        card: str,
        instance: str,
        triton_device_id: int,
        slab_wait: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.prefix = region_prefix(card, instance)
        self.name = self.prefix + process_identity()
        self.nbytes = nbytes
        self.triton_device_id = triton_device_id
        self.slab_wait = slab_wait
        self.slab = SlabAllocator(nbytes)
        self._region_factory = region_factory
        self._region: DeviceRegion | None = None
        self._view: Any = None
        self._lock = threading.Lock()  # single flight: one registration at a time
        self.registered = False
        self.purged = False

    # -- lifecycle ---------------------------------------------------------------------------

    def start(self) -> None:
        """Create the region; :meth:`check` purges this prefix's leftovers and registers it as
        soon as the model server answers (it may start after the engine)."""
        self._region = self._region_factory(self.name, self.nbytes)
        self._view = self._region.view()
        self.metrics.watch_slab(self.slab)
        self.check()

    def purge(self) -> int:
        """Unregister every region the server holds under this card and instance's prefix."""
        status = self.clients[0].get_cuda_shared_memory_status(
            as_json=True, client_timeout=self.timeout
        )
        names = [n for n in (status or {}).get('regions', {}) if n.startswith(self.prefix)]
        for name in names:
            logger.warning('unregistering leftover region %s', name)
            self.clients[0].unregister_cuda_shared_memory(name, client_timeout=self.timeout)
        return len(names)

    def _register(self, cause: str) -> None:
        if self._region is None:
            raise RuntimeError('region not created')
        client = self.clients[0]
        try:
            client.unregister_cuda_shared_memory(self.name, client_timeout=self.timeout)
        except InferenceServerException as exc:  # not registered (the usual case) or down
            logger.debug('unregister before register of %s: %s', self.name, exc)
        client.register_cuda_shared_memory(
            self.name,
            self._region.raw_handle,
            self.triton_device_id,
            self.nbytes,
            client_timeout=self.timeout,
        )
        self.generation += 1
        self.registered = True
        self.server_ready = True
        self.metrics.registrations.labels(cause).inc()
        logger.info('registered region %s (%d bytes, cause %s)', self.name, self.nbytes, cause)

    def reregister(self, seen_generation: int, cause: str) -> None:
        """Register again unless another thread already did since ``seen_generation``."""
        with self._lock:
            if self.generation != seen_generation:
                return
            self.registered = False
            self._register(cause)

    def check(self) -> bool:
        """Health tick: the server answers and holds our region (re-registered when lost)."""
        if not self.check_server():
            self.registered = False
            return False
        try:
            if not self.purged:
                with self._lock:
                    self.purge()
                    self._register('start')
                    self.purged = True
                return self.ready()
            status = self.clients[0].get_cuda_shared_memory_status(
                as_json=True, client_timeout=self.timeout
            )
            present = self.name in (status or {}).get('regions', {})
            if not present:
                self.reregister(self.generation, 'server_lost_region')
        except InferenceServerException as exc:  # the server went away between the calls
            logger.warning('region check failed: %s', exc)
            self.registered = False
            return False
        return self.ready()

    def ready(self) -> bool:
        return self.server_ready and self.registered

    def close(self) -> bool:
        """Unregister, then free. Returns whether the memory was freed here."""
        with self._lock:
            unregistered = False
            if self.registered:
                try:
                    self.clients[0].unregister_cuda_shared_memory(
                        self.name, client_timeout=self.timeout
                    )
                    unregistered = True
                except InferenceServerException as exc:
                    logger.error(
                        'unregister of %s failed (%s): memory left to process exit', self.name, exc
                    )
            else:
                unregistered = True  # the server holds no import of it
            self.registered = False
            self._view = None
            if unregistered and self._region is not None:
                self._region.free()
                self._region = None
                return True
            return False

    # -- per input ---------------------------------------------------------------------------

    def stage(self, nbytes: int) -> Staged:
        try:
            off = self.slab.alloc(nbytes, self.slab_wait)
        except SlabTimeoutError as exc:
            raise OverloadError('slab', str(exc)) from exc
        return Staged(device=self._view[off : off + nbytes], nbytes=nbytes, offset=off)

    @staticmethod
    def commit(staged: Sequence[Staged], sync: Callable[[], None]) -> None:
        """One wait for the stream: every kernel writing these slices finishes before the server
        reads them."""
        sync()
        for s in staged:
            s.committed = True

    def collect(self, p: Pending) -> dict[str, np.ndarray]:
        try:
            return self._wait(p)
        except ModelCallError as exc:
            if _REGION_UNKNOWN not in str(exc):
                raise
        logger.warning('region %s unknown to the server: registering again', self.name)
        try:
            self.reregister(p.generation, 'region_unknown')
        except InferenceServerException as reg:  # the server went away again
            self.registered = False
            raise self._unavailable(p.model, _code(reg), 'unavailable', _message(reg)) from reg
        self._send(p)  # once, from the same slice, by the same request deadline
        return self._wait(p)

    def _tensor(self, staged: Staged, tensor: ModelInput) -> Any:
        inp = self._input(tensor.name, list(tensor.shape), tensor.datatype)
        inp.set_shared_memory(self.name, tensor.nbytes, offset=staged.offset)
        return inp

    def release(self, staged: Staged, sync: Callable[[], None]) -> None:
        if not staged.committed:
            sync()  # a failed request may still have kernels writing its slice
        staged.device = None
        self.slab.release(staged.offset, staged.nbytes)


class CshmRegion:
    """A device allocation exported through ``tritonclient.utils.cuda_shared_memory``."""

    def __init__(self, name: str, nbytes: int, device: int) -> None:
        import tritonclient.utils.cuda_shared_memory as cshm  # needs cuda-python

        self._cshm = cshm
        self._handle: Any = cshm.create_shared_memory_region(name, nbytes, device)
        self.nbytes = nbytes
        self.raw_handle: bytes = cshm.get_raw_handle(self._handle)
        base64.b64decode(self.raw_handle, validate=True)  # fail at start, not at register

    def view(self) -> Any:
        import cupy as cp  # only the engine image has CuPy

        return cp.from_dlpack(
            self._cshm.as_shared_memory_tensor(self._handle, 'UINT8', [self.nbytes])
        )

    def free(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            # drops the module's reference; the region object's finalizer calls cudaFree
            self._cshm.destroy_shared_memory_region(handle)
            del handle
