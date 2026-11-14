"""The closed vocabulary of decode routes: how one photo's pixels were decoded.

``gpu`` is the frame engine's decode (``frame_engine/``). Every other name starts with ``cpu_`` and
says why the photo took the CPU path: the gate in the API (``small``, ``not_image``), the GPU
decode being off or not ready for the profile (``gpu_off``), a call that raised
(``call_failed``), or one of the statuses the model itself answers (:data:`PREP_STATUSES`,
from ``frame_engine/prep_geometry.py`` ``route`` and ``frame_engine/engine.py``). ``cpu_single`` is
the one-image ingest, which never asks the GPU decode.

The Prometheus counter ``op_decode_route_total``, the images document's ``decode_route`` and the
run tallies all use these names; :func:`prep_status_route` rejects a status outside the list.
"""

from __future__ import annotations


GPU = 'gpu'
CPU_SMALL = 'cpu_small'
CPU_NOT_IMAGE = 'cpu_not_image'
CPU_GPU_OFF = 'cpu_gpu_off'
CPU_CALL_FAILED = 'cpu_call_failed'
CPU_SINGLE = 'cpu_single'
CPU_BOX_COUNT = 'cpu_box_count'

#: What the prep model answers instead of ``ok`` (``small`` is also the API gate's own name).
PREP_STATUSES: tuple[str, ...] = (
    'not_jpeg',
    'corrupt',
    'precision',
    'components',
    'progressive',
    'truncated',
    'small',
    'huge',
    'over_budget',
    'gpu_memory',
    'decoder',
)

DECODE_ROUTES: tuple[str, ...] = (
    GPU,
    CPU_NOT_IMAGE,
    CPU_GPU_OFF,
    CPU_CALL_FAILED,
    CPU_SINGLE,
    CPU_BOX_COUNT,
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
