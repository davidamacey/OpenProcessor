"""The closed vocabulary of decode routes: how one photo's pixels were decoded.

``gpu`` is the frame engine's nvJPEG decode (``frame_engine/``). Every other name starts with
``cpu_`` and says why the photo was decoded on a CPU. Two places decode on the CPU:

* inside the frame engine (``frame_engine/host_decode.py``), for an image nvJPEG does not take: the
  reason is one of :data:`CPU_REASONS` (``frame_engine/prep_geometry.py`` ``route``), and the
  engine's answer carries ``cpu_<reason>`` as its ``decode_route``;
* in the API, by the host stage (``cpu_host``): a project with a secondary detector or a letterbox
  fill that is not one grey, a deployment with ``OP_INGEST_GPU_DECODE=0``, a dataset import. A
  photo the engine cannot answer is not decoded anywhere: it fails (``engine_unavailable``,
  ``decode_failed``) and carries no route. The embed reprocess pass still falls back to the API
  per photo (``call_failed``, ``box_count``, :data:`PREP_STATUSES`).

The Prometheus counter ``op_decode_route_total``, the images document's ``decode_route`` and the
run tallies all use these names; :func:`prep_status_route` and :func:`cpu_reason_route` reject a
name outside the list.
"""

from __future__ import annotations


GPU = 'gpu'
CPU_HOST = 'cpu_host'
CPU_CALL_FAILED = 'cpu_call_failed'
CPU_BOX_COUNT = 'cpu_box_count'

#: Why the engine decodes an image on the host (the ``reason`` of a ``cpu`` route in
#: ``frame_engine/prep_geometry.py``).
CPU_REASONS: tuple[str, ...] = (
    'not_jpeg',
    'precision',
    'components',
    'progressive',
    'truncated',
    'small',
)

#: What the engine answers instead of ``ok``: it decoded nothing.
PREP_STATUSES: tuple[str, ...] = (
    'corrupt',
    'huge',
    'over_budget',
    'gpu_memory',
    'decoder',
)

DECODE_ROUTES: tuple[str, ...] = (
    GPU,
    CPU_HOST,
    CPU_CALL_FAILED,
    CPU_BOX_COUNT,
    *(f'cpu_{reason}' for reason in CPU_REASONS),
    *(f'cpu_{status}' for status in PREP_STATUSES),
)


def check_route(route: str) -> str:
    """``route`` itself; raises ``ValueError`` for a name outside :data:`DECODE_ROUTES`."""
    if route not in DECODE_ROUTES:
        raise ValueError(f'unknown decode route {route!r}; the closed list is {DECODE_ROUTES}')
    return route


def prep_status_route(status: str) -> str:
    """The route of a photo the prep model answered with ``status`` (``call_failed`` is the
    API's own word for a call that raised)."""
    return check_route(f'cpu_{status}')


def cpu_reason_route(reason: str) -> str:
    """The ``decode_route`` of an image the engine decoded on the host for ``reason``."""
    if reason not in CPU_REASONS:
        raise ValueError(
            f'unknown decode route reason {reason!r}; the closed list is {CPU_REASONS}'
        )
    return f'cpu_{reason}'
