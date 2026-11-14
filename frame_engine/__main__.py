"""``python -m frame_engine``: start, serve, drain on SIGTERM, then unregister and free.

Order of a stop (SIGTERM or SIGINT): readiness turns false and new requests are refused
(UNAVAILABLE), requests in flight get ``OP_FRAME_ENGINE_DRAIN_SECONDS`` to finish (grpc.aio
``Server.stop(grace)``), the worker threads finish their current photo (every wait inside one is
bounded), and only then is the region unregistered from the model server and freed
(:meth:`frame_engine.transport.CudaShmTransport.close`).
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
from frame_engine.transport import BodyTransport, CshmRegion, CudaShmTransport


MIB = 1024 * 1024
DEVICE = 0  # the one card the container sees
logger = logging.getLogger('frame_engine')


def _transport(s: EngineSettings, metrics: EngineMetrics) -> BodyTransport | CudaShmTransport:
    options = [
        ('grpc.max_send_message_length', 512 * MIB),
        ('grpc.max_receive_message_length', 512 * MIB),
    ]
    clients = [
        grpcclient.InferenceServerClient(s.triton_url, channel_args=options)
        for _ in range(s.triton_channels)
    ]
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


async def _health(transport: Any, every: float, stop: asyncio.Event) -> None:
    """Readiness and re-registration without traffic (a model-server restart)."""
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(1, thread_name_prefix='fe-health') as pool:
        while not stop.is_set():
            await loop.run_in_executor(pool, transport.check)
            try:
                await asyncio.wait_for(stop.wait(), timeout=every)
            except TimeoutError:
                continue


async def serve(s: EngineSettings) -> None:
    metrics = EngineMetrics()
    start_http_server(s.metrics_port, registry=metrics.registry)
    cp.cuda.Device(DEVICE).use()
    transport = _transport(s, metrics)
    transport.start()
    engine = FrameEngine(s, transport, metrics, device=DEVICE)
    service = FrontService(
        engine,
        transport.ready,
        metrics,
        max_in_flight=s.max_in_flight,
        request_deadline=s.request_deadline_seconds,
    )
    server, _port = make_server(
        service, address=f'0.0.0.0:{s.port}', max_message_bytes=s.max_message_mib * MIB
    )
    await server.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    health = asyncio.create_task(_health(transport, s.health_seconds, stop))
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
    freed = transport.close() if isinstance(transport, CudaShmTransport) else True
    logger.info('stopped (region freed here: %s)', freed)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )
    asyncio.run(serve(EngineSettings.from_env()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
