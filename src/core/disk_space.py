"""Free disk space on the volumes the API writes to: the readiness report and the write guard.

``shutil.disk_usage`` gives total, used and free bytes of the filesystem holding a path
(https://docs.python.org/3/library/shutil.html#shutil.disk_usage). A volume is ``low`` below
``OP_DISK_LOW_PCT`` free (a warning) and ``critical`` below ``OP_DISK_CRITICAL_PCT``; at
critical the routes that start ingest, import or upload refuse with 507 instead of failing
halfway through a write.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from typing import Literal

from fastapi import Depends

from src.config.env import env_float
from src.config.projects import projects_data_root
from src.core.exceptions import RequestRejectedError
from src.core.metrics import DISK_FREE_RATIO


logger = logging.getLogger(__name__)

Level = Literal['ok', 'low', 'critical']
DEFAULT_LOW_PCT = 10.0
DEFAULT_CRITICAL_PCT = 3.0
_ORDER: tuple[Level, ...] = ('ok', 'low', 'critical')


@dataclass(frozen=True)
class VolumeSpace:
    volume: str
    free_ratio: float
    level: Level


def thresholds() -> tuple[float, float]:
    """(low, critical) as fractions of the volume."""
    low = env_float('OP_DISK_LOW_PCT', DEFAULT_LOW_PCT, minimum=0.0, maximum=100.0)
    critical = env_float('OP_DISK_CRITICAL_PCT', DEFAULT_CRITICAL_PCT, minimum=0.0, maximum=100.0)
    if critical > low:
        msg = f'OP_DISK_CRITICAL_PCT ({critical}) must not exceed OP_DISK_LOW_PCT ({low})'
        raise ValueError(msg)
    return low / 100.0, critical / 100.0


def classify(free_ratio: float) -> Level:
    low, critical = thresholds()
    if free_ratio < critical:
        return 'critical'
    if free_ratio < low:
        return 'low'
    return 'ok'


def measure(volume: str, free_ratio: float) -> VolumeSpace:
    """Record the gauge, log a warning below ``low``, and return the verdict."""
    level = classify(free_ratio)
    DISK_FREE_RATIO.labels(volume=volume).set(free_ratio)
    if level != 'ok':
        logger.warning('disk %s on volume %s: %.1f%% free', level, volume, free_ratio * 100)
    return VolumeSpace(volume, free_ratio, level)


def data_volume() -> VolumeSpace:
    # A data root that has not been created yet is measured on the filesystem that will hold it.
    path = projects_data_root().absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    usage = shutil.disk_usage(path)
    return measure('data', usage.free / usage.total)


def worst(volumes: list[VolumeSpace]) -> Level:
    return max((v.level for v in volumes), key=_ORDER.index, default='ok')


async def require_disk_space() -> None:
    """Route dependency for ingest, import and upload starts: refuse at ``critical``."""
    space = data_volume()
    if space.level == 'critical':
        raise RequestRejectedError(
            507,
            {
                'error': 'insufficient_storage',
                'message': f'the {space.volume} volume has {space.free_ratio * 100:.1f}% free; '
                'free space before starting new writes',
                'volume': space.volume,
            },
        )


DISK_GUARD = Depends(require_disk_space)
