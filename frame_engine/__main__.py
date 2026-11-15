"""``python -m frame_engine``: start, serve, drain on SIGTERM, then unregister and free.

Order of a stop (SIGTERM or SIGINT): readiness turns false and new requests are refused
(UNAVAILABLE), requests in flight get ``OP_FRAME_ENGINE_DRAIN_SECONDS`` to finish (grpc.aio
``Server.stop(grace)``), the worker threads finish their current photo (every wait inside one is
bounded), and only then is the region unregistered from the model server and freed
(:meth:`frame_engine.transport.CudaShmTransport.close`).

A lost CUDA context (:mod:`frame_engine.cuda_fault`) turns readiness false at once; the health tick
then stops the server the same way and the process exits 1, so the container's restart policy
starts a fresh one. The region is not freed then (CUDA calls fail in a lost context): the process
exit releases it and the next process unregisters it from the model server by its prefix.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cupy as cp
import tritonclient.grpc as grpcclient
from prometheus_client import start_http_server

from frame_engine.engine import FrameEngine
from frame_engine.front import FrontService, make_server
from frame_engine.settings import EngineSettings
from frame_engine.stage_metrics import EngineMetrics
from frame_engine.transport import BodyTransport, CshmRegion, CudaShmTransport, channels


MIB = 1024 * 1024
DEVICE = 0  # the one card the container sees
logger = logging.getLogger('frame_engine')


def _transport(s: EngineSettings, metrics: EngineMetrics) -> BodyTransport | CudaShmTransport:
    clients = channels(s.triton_url, s.triton_channels)
    common: dict[str, Any] = {
        'timeout': s.triton_timeout_seconds,
        'input_factory': grpcclient.InferInput,
        'output_factory': grpcclient.InferRequestedOutput,
    }
    if s.transport == 'body':
        return BodyTransport(clients, metrics, empty=lambda n: cp.empty(n, cp.uint8), **common)
    return CudaShmTransport(
        clients,
        metrics,
        region_factory=lambda name, n: CshmRegion(name, n, DEVICE),
        nbytes=s.region_mib * MIB,
        card=cp.cuda.Device(DEVICE).pci_bus_id,
        instance=s.instance,
        triton_device_id=s.triton_device_id,
        slab_wait=s.slab_wait_seconds,
        **common,
    )


async def _health(
    transport: Any, engine: FrameEngine, every: float, stop: asyncio.Event, wake: asyncio.Event
) -> None:
    """Readiness and re-registration without traffic (a model-server restart), every ``every``
    seconds and at once when a call failed (``wake``, set through ``transport.on_suspect``); a lost
    CUDA context stops the process. ``wake`` is also set by a stop, so the wait ends at once."""
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(1, thread_name_prefix='fe-health') as pool:
        while not stop.is_set():
            if engine.lost.is_set():
                stop.set()
                return
            wake.clear()  # a failure during the probe below asks for another one
            await loop.run_in_executor(pool, transport.check)
            try:
                await asyncio.wait_for(wake.wait(), timeout=every)
            except TimeoutError:
                continue


async def serve(s: EngineSettings) -> int:
    """Serve until SIGTERM or SIGINT (exit 0) or a lost CUDA context (exit 1)."""
    metrics = EngineMetrics()
    start_http_server(s.metrics_port, registry=metrics.registry)
    cp.cuda.Device(DEVICE).use()
    transport = _transport(s, metrics)
    transport.start()
    engine = FrameEngine(s, transport, metrics, device=DEVICE)
    service = FrontService(
        engine,
        lambda: transport.ready() and not engine.lost.is_set(),
        metrics,
        max_in_flight=s.max_in_flight,
        request_deadline=s.request_deadline_seconds,
    )
    server, _port = make_server(
        service, address=f'0.0.0.0:{s.port}', max_message_bytes=s.max_message_mib * MIB
    )
    await server.start()
    stop = asyncio.Event()
    wake = asyncio.Event()
    loop = asyncio.get_running_loop()

    def stopping() -> None:
        stop.set()
        wake.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopping)

    def suspect() -> None:  # from a gRPC callback or worker thread
        loop.call_soon_threadsafe(wake.set)

    transport.on_suspect = suspect
    health = asyncio.create_task(_health(transport, engine, s.health_seconds, stop, wake))
    logger.info(
        'frame engine serving on %d (transport %s, model server %s)',
        s.port,
        s.transport,
        s.triton_url,
    )
    await stop.wait()
    logger.info('stopping: draining %d requests in flight', service.in_flight)
    service.draining = True
    await server.stop(grace=s.drain_seconds)
    await health
    engine.close()
    if engine.lost.is_set():
        logger.critical('stopped after a lost CUDA context: exiting 1 for a restart')
        return 1
    freed = transport.close() if isinstance(transport, CudaShmTransport) else True
    logger.info('stopped (region freed here: %s)', freed)
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )
    settings = EngineSettings.from_env()
    settings.apply_to_interpreter()
    return asyncio.run(serve(settings))


if __name__ == '__main__':
    raise SystemExit(main())
