"""Every setting of the frame engine, read once from ``OP_FRAME_ENGINE_*`` (``env.template``).

Model names and input recipes are not settings: they come with each request's plan. A value that
is set but malformed or out of range refuses startup (the API's ``src/config/env.py`` rule; that
module is not imported because the engine image carries no ``src.config``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


Transport = Literal['cuda', 'body']
TRANSPORTS: tuple[Transport, ...] = ('cuda', 'body')


class SettingsError(ValueError):
    pass


def env_raw(name: str) -> str | None:
    value = os.environ.get(name, '').strip()
    return value or None


def env_num(name: str, default: float, lo: float, hi: float, *, integer: bool) -> float:
    raw = env_raw(name)
    if raw is None:
        return default
    try:
        value = int(raw) if integer else float(raw)
    except ValueError:
        raise SettingsError(f'{name}={raw!r}: not a number') from None
    if not lo <= value <= hi:
        raise SettingsError(f'{name}={raw!r}: outside [{lo:g}, {hi:g}]')
    return value


def env_int(name: str, default: int, lo: int, hi: int) -> int:
    return int(env_num(name, default, lo, hi, integer=True))


def env_float(name: str, default: float, lo: float, hi: float) -> float:
    return env_num(name, default, lo, hi, integer=False)


def env_bool(name: str, default: bool) -> bool:
    raw = env_raw(name)
    if raw is None:
        return default
    if raw.lower() in ('1', 'true', 'yes', 'on'):
        return True
    if raw.lower() in ('0', 'false', 'no', 'off'):
        return False
    raise SettingsError(f'{name}={raw!r}: not a boolean')


@dataclass(frozen=True)
class EngineSettings:
    triton_url: str = 'triton-server:8001'
    transport: Transport = 'cuda'
    #: Size of the one CUDA shared-memory region (every model input in flight is a slice of it).
    region_mib: int = 1024
    #: Names this process's region with the card: one engine slot per card by default.
    instance: str = 'fe0'
    #: The card's device id as the model server numbers its devices (0 when it sees one card).
    triton_device_id: int = 0
    triton_channels: int = 4
    triton_timeout_seconds: float = 60.0
    workers: int = 16
    lanes: int = 16
    encode_threads: int = 8
    #: Requests admitted at once (running or queued for a worker); above it: overload.
    max_in_flight: int = 128
    pool_mib: int = 4096
    budget_mib: int = 2048
    #: Bounded waits; a timeout is the overload answer. The API retries an overload three times,
    #: so four budget waits stay inside the 20 s the Triton model waited before it gave up.
    budget_wait_seconds: float = 4.0
    slab_wait_seconds: float = 4.0
    min_megapixels: float = 0.0
    max_megapixels: float = 100.0
    jpeg_quality: int = 90
    #: Long-side cap of the crop JPEGs in pixels (0 = source size); keep equal to OP_CROP_MAX_SIDE.
    crop_max_side: int = 0
    #: Wait for the stream at the end of a kernel stage so its GPU time lands there (attribution).
    stage_sync: bool = False
    port: int = 8001
    metrics_port: int = 8002
    max_message_mib: int = 256
    drain_seconds: float = 30.0
    health_seconds: float = 2.0

    @classmethod
    def from_env(cls) -> EngineSettings:
        d = cls()
        transport = env_raw('OP_FRAME_ENGINE_TRANSPORT') or d.transport
        if transport not in TRANSPORTS:
            raise SettingsError(
                f'OP_FRAME_ENGINE_TRANSPORT={transport!r}: one of {", ".join(TRANSPORTS)}'
            )
        instance = env_raw('OP_FRAME_ENGINE_INSTANCE') or d.instance
        if not instance.replace('-', '').isalnum():
            raise SettingsError(
                f'OP_FRAME_ENGINE_INSTANCE={instance!r}: letters, digits and dashes only'
            )
        s = cls(
            triton_url=env_raw('OP_FRAME_ENGINE_TRITON_URL') or d.triton_url,
            transport=transport,
            region_mib=env_int('OP_FRAME_ENGINE_REGION_MIB', d.region_mib, 16, 16384),
            instance=instance,
            triton_device_id=env_int('OP_FRAME_ENGINE_TRITON_DEVICE_ID', d.triton_device_id, 0, 63),
            triton_channels=env_int('OP_FRAME_ENGINE_TRITON_CHANNELS', d.triton_channels, 1, 64),
            triton_timeout_seconds=env_float(
                'OP_FRAME_ENGINE_TRITON_TIMEOUT_SECONDS', d.triton_timeout_seconds, 1, 600
            ),
            workers=env_int('OP_FRAME_ENGINE_WORKERS', d.workers, 1, 256),
            lanes=env_int('OP_FRAME_ENGINE_LANES', d.lanes, 1, 256),
            encode_threads=env_int('OP_FRAME_ENGINE_ENCODE_THREADS', d.encode_threads, 1, 256),
            max_in_flight=env_int('OP_FRAME_ENGINE_MAX_IN_FLIGHT', d.max_in_flight, 1, 100_000),
            pool_mib=env_int('OP_FRAME_ENGINE_POOL_MIB', d.pool_mib, 64, 262_144),
            budget_mib=env_int('OP_FRAME_ENGINE_BUDGET_MIB', d.budget_mib, 64, 262_144),
            budget_wait_seconds=env_float(
                'OP_FRAME_ENGINE_BUDGET_WAIT_SECONDS', d.budget_wait_seconds, 0, 600
            ),
            slab_wait_seconds=env_float(
                'OP_FRAME_ENGINE_SLAB_WAIT_SECONDS', d.slab_wait_seconds, 0, 600
            ),
            min_megapixels=env_float('OP_FRAME_ENGINE_MIN_MEGAPIXELS', d.min_megapixels, 0, 10_000),
            max_megapixels=env_float('OP_FRAME_ENGINE_MAX_MEGAPIXELS', d.max_megapixels, 0, 10_000),
            jpeg_quality=env_int('OP_FRAME_ENGINE_JPEG_QUALITY', d.jpeg_quality, 1, 100),
            crop_max_side=env_int('OP_FRAME_ENGINE_CROP_MAX_SIDE', d.crop_max_side, 0, 65_535),
            stage_sync=env_bool('OP_FRAME_ENGINE_STAGE_SYNC', d.stage_sync),
            port=env_int('OP_FRAME_ENGINE_PORT', d.port, 1, 65_535),
            metrics_port=env_int('OP_FRAME_ENGINE_METRICS_PORT', d.metrics_port, 1, 65_535),
            max_message_mib=env_int('OP_FRAME_ENGINE_MAX_MESSAGE_MIB', d.max_message_mib, 1, 2047),
            drain_seconds=env_float('OP_FRAME_ENGINE_DRAIN_SECONDS', d.drain_seconds, 0, 3600),
            health_seconds=env_float('OP_FRAME_ENGINE_HEALTH_SECONDS', d.health_seconds, 0.1, 600),
        )
        if s.budget_mib > s.pool_mib:
            raise SettingsError(
                f'OP_FRAME_ENGINE_BUDGET_MIB {s.budget_mib} exceeds '
                f'OP_FRAME_ENGINE_POOL_MIB {s.pool_mib}'
            )
        return s
