"""How a model input reaches Triton: one CUDA shared-memory region, or the plain request body.

The engine calls the TensorRT models as a Triton gRPC client. A photo's model inputs are written
by the kernels into a device buffer (:meth:`Transport.stage`), made visible to the model server
(:meth:`Transport.commit`), sent (:meth:`Transport.infer`) and given back
(:meth:`Transport.release`). Two implementations, chosen by ``OP_FRAME_ENGINE_TRANSPORT``:

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
  registrations) re-registers once under a single-flight lock and retries the call once. The health
  check (:meth:`CudaShmTransport.check`) re-registers too when the server is back and the region is
  missing, so readiness returns without traffic.
* Shutdown unregisters first, then frees: the exporting process must outlive every import (CUDA
  documents freeing exported memory before the importer closes it as undefined behaviour). The
  memory is freed only after a successful unregister; otherwise the process exit releases it.
* The slab wait is bounded; a timeout is :class:`OverloadError`.

Every model call has a deadline: ``OP_FRAME_ENGINE_TRITON_TIMEOUT_SECONDS``, or what is left of the
request's deadline when that is sooner. The call is issued with tritonclient's ``async_infer`` (the
gRPC deadline is its ``client_timeout``) and the worker waits for the answer no longer than that
deadline, then cancels the call (``CallContext.cancel``): a channel lost with the call in flight
cannot hold a worker, its slice or its budget. A call without an answer in time, a channel or server
that went away (UNAVAILABLE, CANCELLED) and a call the client library itself fails are
:class:`TritonUnavailableError` (the request is answered UNAVAILABLE, retryable); an error the
server answered is :class:`ModelCallError` (the branch fails). Each failure is counted with its
reason in ``op_frame_engine_triton_call_errors_total``.
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
from tritonclient.utils import InferenceServerException

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


@dataclass
class Staged:
    """One staged input: a flat uint8 device array and how it reaches the server."""

    device: Any
    nbytes: int
    offset: int = -1  # slice of the region (cuda); -1 for the body transport
    host: np.ndarray | None = None  # the body transport's host copy, after commit


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

    def _call(self, model: str, inp: Any, output: str, deadline: float) -> np.ndarray:
        """One model call that ends by its deadline, answered or not (module docstring)."""
        timeout = min(self.timeout, deadline - time.perf_counter())
        if timeout <= 0:
            raise self._unavailable(model, _STATUS_DEADLINE, 'deadline', 'request deadline passed')
        answer = _Answer()
        try:
            call = self._client().async_infer(
                model, [inp], answer, outputs=[self._output(output)], client_timeout=timeout
            )
        except InferenceServerException as exc:
            raise self._failed(model, exc) from exc
        except Exception as exc:  # the client refused the call itself (a closed channel)
            logger.warning('%s: the client failed the call', model, exc_info=True)
            raise self._unavailable(model, type(exc).__name__, 'client', str(exc)) from exc
        if not answer.done.wait(timeout):
            call.cancel()
            raise self._unavailable(
                model, _STATUS_DEADLINE, 'deadline', f'no answer within {timeout:.2f} s'
            )
        if answer.error is not None:
            raise self._failed(model, answer.error) from answer.error
        out: np.ndarray = answer.result.as_numpy(output)
        return out

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

    def commit(self, staged: Staged) -> None:
        staged.host = staged.device.get()

    def infer(
        self,
        model: str,
        staged: Staged,
        byte_offset: int,
        shape: list[int],
        output: str,
        *,
        deadline: float = math.inf,
    ) -> np.ndarray:
        if staged.host is None:
            raise RuntimeError('infer before commit')
        n = int(np.prod(shape))
        inp = self._input('images', shape, 'UINT8')
        inp.set_data_from_numpy(staged.host[byte_offset : byte_offset + n].reshape(shape))
        return self._call(model, inp, output, deadline)

    def release(self, staged: Staged, sync: Callable[[], None]) -> None:
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
        self.generation = 0  # bumped by every successful registration
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

    def commit(self, _staged: Staged) -> None:
        """The kernels writing the slice finish before the server reads it."""
        import cupy as cp  # the offline suite runs the transport without CuPy

        cp.cuda.get_current_stream().synchronize()

    def infer(
        self,
        model: str,
        staged: Staged,
        byte_offset: int,
        shape: list[int],
        output: str,
        *,
        deadline: float = math.inf,
    ) -> np.ndarray:
        n = int(np.prod(shape))
        gen = self.generation
        try:
            return self._call(
                model, self._shm_input(staged, byte_offset, shape, n), output, deadline
            )
        except ModelCallError as exc:
            if _REGION_UNKNOWN not in str(exc):
                raise
            logger.warning('region %s unknown to the server: registering again', self.name)
            try:
                self.reregister(gen, 'region_unknown')
            except InferenceServerException as reg:  # the server went away again
                self.registered = False
                raise self._unavailable(model, _code(reg), 'unavailable', _message(reg)) from reg
            return self._call(
                model, self._shm_input(staged, byte_offset, shape, n), output, deadline
            )

    def _shm_input(self, staged: Staged, byte_offset: int, shape: list[int], n: int) -> Any:
        inp = self._input('images', shape, 'UINT8')
        inp.set_shared_memory(self.name, n, offset=staged.offset + byte_offset)
        return inp

    def release(self, staged: Staged, sync: Callable[[], None]) -> None:
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
