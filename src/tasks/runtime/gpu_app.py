"""The Celery app factory every image uses, and the worker-start checks.

``build_app`` is the one place the configuration of the job plan (02b section 7.1) is written.
The API image's ``src/tasks/app.py`` calls it; the trainer, export and evaluator images call
``make_app``, which adds the worker signals those images need. Nothing here imports torch,
cuml, tritonclient or opens a database connection at import.
"""

from __future__ import annotations

import logging
import socket
import ssl
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from celery import Celery
from celery.signals import worker_process_init, worker_ready
from kombu import Queue

from src.config.job_kinds import KIND_BY_TASK
from src.config.queue import NON_JOB_QUEUES, QueueSettings, TaskQueues, broker_transport_options
from src.config.task_routes import ROUTES, celery_task_routes


if TYPE_CHECKING:
    from collections.abc import Sequence


APP_NAME = 'openprocessor'
DEFAULT_SOFT_TIME_LIMIT_S = 10800
DEFAULT_TIME_LIMIT_S = 11700


def static_config(include: Sequence[str]) -> dict[str, Any]:
    """Every setting that does not need the environment."""
    return {
        'include': list(include),
        'task_queues': [Queue(name) for name in TaskQueues.ALL],
        'task_create_missing_queues': False,
        'task_routes': celery_task_routes(),
        'task_serializer': 'json',
        'result_serializer': 'json',
        'event_serializer': 'json',
        'accept_content': ['json'],
        'result_backend': None,
        'task_ignore_result': True,
        'task_track_started': False,
        'task_acks_late': True,
        'task_retry_on_worker_lost': True,
        'worker_prefetch_multiplier': 1,
        'broker_transport_options': broker_transport_options(),
        'broker_connection_retry_on_startup': True,
        'broker_connection_max_retries': None,
        'task_soft_time_limit': DEFAULT_SOFT_TIME_LIMIT_S,
        'task_time_limit': DEFAULT_TIME_LIMIT_S,
        'worker_proc_alive_timeout': 30,
        'worker_soft_shutdown_timeout': 30,
        'worker_enable_soft_shutdown_on_idle': True,
        'worker_cancel_long_running_tasks_on_connection_loss': False,
        'worker_send_task_events': True,
        'task_send_sent_event': True,
        'enable_utc': True,
        'timezone': 'UTC',
    }


def _apply_broker(source: Any, **_: Any) -> None:
    """Read the queue URL and password when the configuration loads, not at import."""
    settings = QueueSettings.from_env()
    source['broker_url'] = settings.broker_url
    if settings.uses_tls:
        source['broker_use_ssl'] = {'ssl_cert_reqs': ssl.CERT_REQUIRED}


def build_app(include: Sequence[str]) -> Celery:
    """A Celery app with the plan's configuration. The configuration loads lazily, on first use."""
    app = Celery(APP_NAME, set_as_current=False)
    app.config_from_object(static_config(include))
    app.on_after_configure.connect(_apply_broker, sender=app, weak=False)
    return app


def make_app(include: Sequence[str], device_uuid: str | None = None) -> Celery:
    """The app of a GPU image: ``include`` lists its handler modules.

    ``device_uuid`` is the single card a device-scoped family's worker is bound to; ``None`` in
    0.7.0 because no family is device-scoped. Whatever the argument, the worker records every
    card its container sees (see :func:`start_inventory`).
    """
    global _gpu_worker_device  # noqa: PLW0603 - the pool child reads what the parent configured
    _gpu_worker_device = _DeviceScope(device_uuid)
    install_worker_signals()
    return build_app(include)


def dispose_inherited_engines() -> None:
    """In a pool child: drop database engines inherited from the parent, closing nothing.

    ``dispose(close=False)`` abandons the parent's connections without closing the sockets the
    parent still uses ([pooling and fork](https://docs.sqlalchemy.org/en/21/core/pooling.html)).
    No I/O: the docs allow the child a few seconds. A parent that never imported the engine
    module has nothing to drop.
    """
    engine_module = sys.modules.get('src.db.engine')
    if engine_module is not None:
        engine_module.dispose_inherited_engines()


def task_route_problems(app: Celery) -> tuple[list[str], list[str]]:
    """Registered tasks with no route, and routed job tasks with no kind."""
    names = sorted(name for name in app.tasks if not name.startswith('celery.'))
    unrouted = [name for name in names if name not in ROUTES]
    kindless = [
        name
        for name in names
        if name in ROUTES and ROUTES[name] not in NON_JOB_QUEUES and name not in KIND_BY_TASK
    ]
    return unrouted, kindless


def validate_task_routes(app: Celery) -> None:
    """Refuse to serve when a registered task has no route or no kind (it would hit 'celery')."""
    unrouted, kindless = task_route_problems(app)
    if unrouted or kindless:
        raise SystemExit(f'unrouted tasks: {unrouted}; tasks without a job kind: {kindless}')


def _on_worker_ready(sender: Any = None, **_: Any) -> None:
    from celery import current_app

    validate_task_routes(getattr(sender, 'app', None) or current_app)


def _on_worker_process_init(**_: Any) -> None:
    dispose_inherited_engines()
    if _gpu_worker_device is not None:
        start_inventory(_gpu_worker_device.uuid)


def install_worker_signals() -> None:
    """Connect the pool-child and ready handlers once (``dispatch_uid`` dedupes repeats)."""
    worker_ready.connect(_on_worker_ready, dispatch_uid='op.worker_ready', weak=False)
    worker_process_init.connect(
        _on_worker_process_init, dispatch_uid='op.worker_process_init', weak=False
    )


# --- the card inventory of a GPU worker (research GPU S-6, S-12; review RR-9) -----------------

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DeviceScope:
    uuid: str | None


@dataclass(frozen=True)
class GpuDevice:
    """One ``gpu_devices`` row, keyed by the NVML UUID (CUDA numbers cards fastest first)."""

    gpu_uuid: str
    host: str
    nvml_index: int
    name: str
    compute_capability: str
    memory_total_mib: int
    numa_node: int | None
    pcie_gen: int | None
    pcie_width: int | None
    hw_jpeg_decoder: bool
    driver_version: str
    cuda_version: str


class InventoryWriter(Protocol):
    """The database side, registered by the job runtime (it needs ``src.db``)."""

    def upsert_devices(self, devices: Sequence[GpuDevice]) -> None: ...

    def touch_devices(self, uuids: Sequence[str]) -> None: ...

    def record_capabilities(self, capabilities: dict[str, Any]) -> None: ...


_gpu_worker_device: _DeviceScope | None = None
_inventory_writer: InventoryWriter | None = None
_inventory: tuple[str, ...] | None = None


def set_inventory_writer(writer: InventoryWriter) -> None:
    """Register the writer that stores ``gpu_devices`` and ``worker_capabilities``."""
    global _inventory_writer  # noqa: PLW0603 - one writer per process, set by the runtime
    _inventory_writer = writer


def _optional(nvml: Any, name: str, handle: Any) -> Any:
    """One NVML query that a card may not support (it raises ``NVMLError``): ``None`` then."""
    try:
        return getattr(nvml, name)(handle)
    except nvml.NVMLError:
        return None


def read_devices(nvml: Any) -> list[GpuDevice]:
    """Every card NVML shows this container. Call it in the child, after fork, never the parent."""
    host = socket.gethostname()
    driver = nvml.nvmlSystemGetDriverVersion()
    cuda_raw = nvml.nvmlSystemGetCudaDriverVersion()
    cuda = f'{cuda_raw // 1000}.{(cuda_raw % 1000) // 10}'
    devices = []
    for index in range(nvml.nvmlDeviceGetCount()):
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        major, minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)
        devices.append(
            GpuDevice(
                gpu_uuid=nvml.nvmlDeviceGetUUID(handle),
                host=host,
                nvml_index=index,
                name=nvml.nvmlDeviceGetName(handle),
                compute_capability=f'{major}.{minor}',
                memory_total_mib=nvml.nvmlDeviceGetMemoryInfo(handle).total // (1024 * 1024),
                numa_node=_optional(nvml, 'nvmlDeviceGetNumaNodeId', handle),
                pcie_gen=_optional(nvml, 'nvmlDeviceGetMaxPcieLinkGeneration', handle),
                pcie_width=_optional(nvml, 'nvmlDeviceGetMaxPcieLinkWidth', handle),
                # NVML answers "not supported" on cards without a JPEG engine.
                hw_jpeg_decoder=_optional(nvml, 'nvmlDeviceGetJpgUtilization', handle) is not None,
                driver_version=driver,
                cuda_version=cuda,
            )
        )
    return devices


def _import_pynvml() -> Any:
    import pynvml  # NVML is initialised in the child only, so the import stays lazy

    return pynvml


def start_inventory(device_uuid: str | None, nvml: Any = None) -> None:
    """Upsert a ``gpu_devices`` row for every visible card and list them in the capabilities.

    ``device_uuid``, when given, must be one of the visible cards. A container with no cards
    records nothing; cards without a registered writer is a wiring error and refuses to start.
    """
    global _inventory  # noqa: PLW0603 - the heartbeat refreshes what the start recorded
    library = nvml if nvml is not None else _import_pynvml()
    library.nvmlInit()
    try:
        devices = read_devices(library)
    finally:
        library.nvmlShutdown()
    if not devices:
        logger.info('gpu_inventory_none_visible')
        return
    uuids = tuple(device.gpu_uuid for device in devices)
    if device_uuid is not None and device_uuid not in uuids:
        raise RuntimeError(f'device {device_uuid} is not visible in this container ({uuids})')
    if _inventory_writer is None:
        raise RuntimeError('GPU cards are visible but no inventory writer is registered')
    _inventory_writer.upsert_devices(devices)
    _inventory_writer.record_capabilities({'gpu_uuids': list(uuids), 'device_uuid': device_uuid})
    _inventory = uuids


def refresh_inventory() -> None:
    """On every job heartbeat: refresh ``seen_at`` so a long-lived worker keeps its rows."""
    if _inventory and _inventory_writer is not None:
        _inventory_writer.touch_devices(_inventory)
