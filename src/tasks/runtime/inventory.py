"""The database side of the GPU card inventory: ``gpu_devices`` and ``worker_capabilities``.

``src.tasks.runtime.gpu_app`` reads the cards and calls a registered writer; this is that writer.
Both tables are deployment data without row-level security, so the upserts need no project
scope (``deployment_tx`` is used because the module is on the allow-list and the session helper
requires a scope). Review RN-8 of the schema plan: it touches only these two tables.
"""

from __future__ import annotations

import json
import socket
from typing import TYPE_CHECKING, Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db.engine import get_sync_engine
from src.db.repos import gpu_devices
from src.db.session import sync_deployment_tx


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.tasks.runtime.gpu_app import GpuDevice

_TOUCH_DEVICES = text('UPDATE gpu_devices SET seen_at = now() WHERE gpu_uuid = ANY(:uuids)')
_UPSERT_CAPABILITIES = text(
    'INSERT INTO worker_capabilities (worker_id, queue, host, capabilities, seen_at) '
    'VALUES (:worker_id, :queue, :host, CAST(:capabilities AS jsonb), now()) '
    'ON CONFLICT (worker_id) DO UPDATE SET queue = excluded.queue, host = excluded.host, '
    'capabilities = excluded.capabilities, seen_at = now()'
)


def worker_id(queue: str) -> str:
    """``<queue>@<hostname>``, the key of this worker's ``worker_capabilities`` row."""
    return f'{queue}@{socket.gethostname()}'


class DbInventoryWriter:
    """Writes the inventory through the child's sync engine."""

    def __init__(self, queue: str) -> None:
        self.queue = queue

    def _session(self) -> Session:
        return Session(get_sync_engine(), expire_on_commit=False)

    def upsert_devices(self, devices: Sequence[GpuDevice]) -> None:
        rows = [
            {
                'gpu_uuid': d.gpu_uuid,
                'host': d.host,
                'nvml_index': d.nvml_index,
                'name': d.name,
                'compute_capability': d.compute_capability,
                'memory_total_mib': d.memory_total_mib,
                'numa_node': d.numa_node,
                'pcie_gen': d.pcie_gen,
                'pcie_width': d.pcie_width,
                'hw_jpeg_decoder': d.hw_jpeg_decoder,
                'driver_version': d.driver_version,
                'cuda_version': d.cuda_version,
            }
            for d in devices
        ]
        with self._session() as session, sync_deployment_tx(session, 'gpu inventory upsert'):
            session.execute(gpu_devices.UPSERT, rows)

    def touch_devices(self, uuids: Sequence[str]) -> None:
        with self._session() as session, sync_deployment_tx(session, 'gpu inventory refresh'):
            session.execute(_TOUCH_DEVICES, {'uuids': list(uuids)})

    def record_capabilities(self, capabilities: dict[str, Any]) -> None:
        with self._session() as session, sync_deployment_tx(session, 'worker capabilities'):
            session.execute(
                _UPSERT_CAPABILITIES,
                {
                    'worker_id': worker_id(self.queue),
                    'queue': self.queue,
                    'host': socket.gethostname(),
                    'capabilities': json.dumps(capabilities),
                },
            )
