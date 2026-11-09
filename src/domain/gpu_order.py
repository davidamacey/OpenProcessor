"""``OP_TRAIN_GPU_ORDER``: which host card sits at which position inside the trainer container.

Docker's ``device_ids`` list pins a container to some host cards; inside it they are numbered
``0..n-1`` in list order (NVML, and so ``gpu_devices.nvml_index`` as that container records it).
A run names *host* card numbers, the ids the arbiter and ``nvidia-smi`` use. Unset means the
container sees every card at its host number. Pure; the card's identity is its UUID, never either
number (review RR-9).
"""

from __future__ import annotations


def parse_order(raw: str | None) -> tuple[int, ...]:
    """``'2,0'`` -> ``(2, 0)``; empty or unset -> ``()`` (unrestricted)."""
    return tuple(int(tok) for tok in (raw or '').split(',') if tok.strip())


def position_of(order: tuple[int, ...], host_id: int) -> int:
    """The position inside the container of host card ``host_id``.

    Raises ``ValueError`` when the container is pinned to a list that does not contain it.
    """
    if not order:
        return host_id
    try:
        return order.index(host_id)
    except ValueError:
        msg = f'host GPU {host_id} is not attached to this container (attached: {order})'
        raise ValueError(msg) from None


def host_id_at(order: tuple[int, ...], position: int) -> int:
    """The host card number at container position ``position`` (identity when unrestricted)."""
    return order[position] if order and position < len(order) else position
