"""``gpu_devices``: the card inventory, keyed by NVML UUID. Deployment data, no row-level security.

The upsert is the one statement every writer uses: the GPU worker at start, and a reservation in
the transaction that reserves the card (review RR-9), so a card no worker ever recorded can still
be held.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select, text

from src.db.models.jobs import GpuDevice


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from sqlalchemy.orm import Session


@dataclass(frozen=True)
class Card:
    """One card as a ``gpu_devices`` row carries it."""

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


UPSERT = text(
    'INSERT INTO gpu_devices (gpu_uuid, host, nvml_index, name, compute_capability, '
    'memory_total_mib, numa_node, pcie_gen, pcie_width, hw_jpeg_decoder, driver_version, '
    'cuda_version, seen_at) VALUES (:gpu_uuid, :host, :nvml_index, :name, :compute_capability, '
    ':memory_total_mib, :numa_node, :pcie_gen, :pcie_width, :hw_jpeg_decoder, :driver_version, '
    ':cuda_version, now()) ON CONFLICT (gpu_uuid) DO UPDATE SET host = excluded.host, '
    'nvml_index = excluded.nvml_index, name = excluded.name, '
    'compute_capability = excluded.compute_capability, '
    'memory_total_mib = excluded.memory_total_mib, numa_node = excluded.numa_node, '
    'pcie_gen = excluded.pcie_gen, pcie_width = excluded.pcie_width, '
    'hw_jpeg_decoder = excluded.hw_jpeg_decoder, driver_version = excluded.driver_version, '
    'cuda_version = excluded.cuda_version, seen_at = now()'
)


def upsert(session: Session, cards: Sequence[Card]) -> None:
    session.execute(UPSERT, [vars(c) for c in cards])


def on_host_at(session: Session, host: str, positions: Sequence[int]) -> list[Card]:
    """The cards of ``host`` at container positions ``positions`` (``nvml_index``)."""
    found = session.execute(
        select(GpuDevice).where(GpuDevice.host == host, GpuDevice.nvml_index.in_(list(positions)))
    )
    return _cards(found.scalars())


def on_host(session: Session, host: str) -> list[Card]:
    """Every card the worker of ``host`` recorded at start (a GPU container sees its own)."""
    return _cards(session.execute(select(GpuDevice).where(GpuDevice.host == host)).scalars())


def at_positions(session: Session, positions: Sequence[int]) -> list[Card]:
    """The cards recorded at ``nvml_index`` in ``positions`` on any host (a CPU-side job names
    host card numbers and has no card inventory of its own)."""
    found = session.execute(select(GpuDevice).where(GpuDevice.nvml_index.in_(list(positions))))
    return _cards(found.scalars())


def _cards(rows: Iterable[GpuDevice]) -> list[Card]:
    return [Card(**{k: getattr(row, k) for k in Card.__dataclass_fields__}) for row in rows]
