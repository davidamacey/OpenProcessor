"""Runtime detection of GPU vs CPU clustering backend.

Probes cuML + cupy availability and free VRAM on every call (~microseconds).
Returns ``'gpu'`` if a usable device with enough free VRAM is present,
``'cpu'`` otherwise. No environment-variable toggle — purely hardware /
library capability.

Per-call detection (not per-process cached) is intentional: the
``make gpu-free`` workflow strips the worker container's GPU access
without restarting the worker. The next recluster's first call here
sees no VRAM and falls back to CPU silently.

Public API:

* :func:`detect_cluster_backend` — returns a :class:`BackendInfo` for the
  current call. Cheap.
* :func:`free_gpu_blocks` — drain cuML/cupy memory pool after each UMAP
  fit; mitigates the known VRAM-creep gotcha (cuml#4068).
* :func:`gpu_used_vram_mb` — best-effort "how much is used right now" for
  the peak-VRAM telemetry on the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.core.logging import get_logger


logger = get_logger(__name__)


# nn-descent UMAP on 90k x 1024 peaks at 2-6 GB; 3 GB floor covers the
# fit plus a small safety margin. brute_force_knn (deterministic) needs
# 8-15 GB — picked 12 GB as the threshold above which we switch to it.
MIN_FREE_VRAM_GB_NN_DESCENT = 3.0
MIN_FREE_VRAM_GB_BRUTE_FORCE = 12.0


ClusterBackendName = Literal['gpu', 'cpu']
UMapBuildAlgo = Literal['nn_descent', 'brute_force_knn']


@dataclass(frozen=True)
class BackendInfo:
    """Result of a backend probe. Surfaced to the dashboard."""

    name: ClusterBackendName
    detail: str
    free_vram_mb: int | None
    build_algo: UMapBuildAlgo | None  # only meaningful when name == 'gpu'


def _try_import_cuml() -> tuple[Any, Any]:
    try:
        import cuml  # type: ignore[import-not-found]
        import cupy  # type: ignore[import-not-found]
    except ImportError:
        return None, None
    return cuml, cupy


def _cpu_detail() -> str:
    bits: list[str] = []
    try:
        import sklearn

        bits.append(f'sklearn {sklearn.__version__}')
    except ImportError:
        pass
    try:
        import umap

        bits.append(f'umap-learn {umap.__version__}')
    except ImportError:
        pass
    return 'cpu (' + ' · '.join(bits or ['stdlib']) + ')'


def _gpu_device_name(cupy: Any) -> tuple[int, str]:
    """Best-effort current-device id + readable name."""
    try:
        device_id = int(cupy.cuda.runtime.getDevice())
        props = cupy.cuda.runtime.getDeviceProperties(device_id)
        raw_name = props.get('name') if isinstance(props, dict) else None
        if isinstance(raw_name, bytes):
            name = raw_name.decode('ascii', errors='replace').strip()
        elif isinstance(raw_name, str):
            name = raw_name.strip()
        else:
            name = 'GPU'
        return device_id, name or 'GPU'
    except Exception:
        return -1, 'GPU'


def detect_cluster_backend() -> BackendInfo:
    """Probe for GPU clustering capability.

    Safe to call on every dispatch — no caching, just a few microseconds
    of cupy calls. Returns a :class:`BackendInfo` populated with the
    chosen backend, a human-readable detail string for the dashboard
    chip, free VRAM at probe time, and (GPU only) the UMAP build
    algorithm that fits in current VRAM.
    """
    cuml, cupy = _try_import_cuml()
    if cuml is None or cupy is None:
        return BackendInfo(
            name='cpu',
            detail=_cpu_detail(),
            free_vram_mb=None,
            build_algo=None,
        )

    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:
        logger.info('legacy_cluster_backend_cuda_unavailable', error=str(exc))
        return BackendInfo(
            name='cpu',
            detail=_cpu_detail(),
            free_vram_mb=None,
            build_algo=None,
        )

    if device_count <= 0:
        return BackendInfo(
            name='cpu',
            detail=_cpu_detail(),
            free_vram_mb=None,
            build_algo=None,
        )

    # Drain the pool so memGetInfo reports the genuine free VRAM and not
    # whatever cupy is holding for itself. Cheap; ignore failures.
    import contextlib

    with contextlib.suppress(Exception):
        cupy.get_default_memory_pool().free_all_blocks()

    try:
        free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
    except Exception as exc:
        logger.warning('legacy_cluster_backend_memgetinfo_failed', error=str(exc))
        return BackendInfo(
            name='cpu',
            detail=_cpu_detail(),
            free_vram_mb=None,
            build_algo=None,
        )

    free_mb = int(free_bytes // (1024 * 1024))
    total_mb = int(total_bytes // (1024 * 1024))

    if free_mb < int(MIN_FREE_VRAM_GB_NN_DESCENT * 1024):
        logger.info(
            'legacy_cluster_backend_low_vram_fallback',
            free_mb=free_mb,
            min_required_mb=int(MIN_FREE_VRAM_GB_NN_DESCENT * 1024),
        )
        return BackendInfo(
            name='cpu',
            detail=_cpu_detail(),
            free_vram_mb=free_mb,
            build_algo=None,
        )

    device_id, device_name = _gpu_device_name(cupy)
    build_algo: UMapBuildAlgo = (
        'brute_force_knn' if free_mb >= int(MIN_FREE_VRAM_GB_BRUTE_FORCE * 1024) else 'nn_descent'
    )

    cuml_ver = getattr(cuml, '__version__', '?')
    detail = (
        f'gpu (cuml {cuml_ver} · {device_name} GPU {device_id} · '
        f'{free_mb}/{total_mb} MB free · {build_algo})'
    )
    return BackendInfo(
        name='gpu',
        detail=detail,
        free_vram_mb=free_mb,
        build_algo=build_algo,
    )


def free_gpu_blocks() -> None:
    """Drain cuML/cupy memory pool.

    Call after every UMAP fit to fight cuml#4068 (VRAM creep across
    fit_transform calls in a long-lived worker process).
    """
    _, cupy = _try_import_cuml()
    if cupy is None:
        return
    try:
        cupy.get_default_memory_pool().free_all_blocks()
    except Exception as exc:
        logger.debug('legacy_cluster_backend_free_blocks_failed', error=str(exc))


def gpu_used_vram_mb() -> int | None:
    """Current VRAM usage on the active device, in MB.

    Returns None when GPU support is unavailable. Used for the peak-VRAM
    telemetry on the dashboard — sample before/after UMAP fit and report
    the max.
    """
    _, cupy = _try_import_cuml()
    if cupy is None:
        return None
    try:
        free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
    except Exception:
        return None
    return int((total_bytes - free_bytes) // (1024 * 1024))


__all__ = [
    'MIN_FREE_VRAM_GB_BRUTE_FORCE',
    'MIN_FREE_VRAM_GB_NN_DESCENT',
    'BackendInfo',
    'ClusterBackendName',
    'UMapBuildAlgo',
    'detect_cluster_backend',
    'free_gpu_blocks',
    'gpu_used_vram_mb',
]
