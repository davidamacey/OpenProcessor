"""Background region-clustering jobs: single-flight runs, cross-worker
status files and the re-partition TTL markers.

The API runs many uvicorn workers in one container; in-process job state
would make a status poll hit a worker that knows nothing about the job, so
state lives in small files under the bound project's state dir. The
algorithms the jobs run live in
:mod:`src.services.curation.clustering.region_box_clustering`.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.services.curation.clustering.region_box_clustering import (
    auto_assign_fp_from_centroids,
    build_region_fp_centroids,
    cluster_region_residuals,
    count_false_positive_boxes,
)


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)


# Background region-clustering job state, persisted to a small file. The API
# runs many uvicorn workers in one container; in-process state would make the
# status poll hit a worker that knows nothing about the job. A file on the
# shared container fs is consistent across all workers. Clustering 50k+
# regions scrolls hundreds of MB + writes every assignment (~7-8 min), far
# too long to hold an HTTP request — so the endpoint fires it and the UI
# polls this file.
def _region_state_dir() -> Path:
    """The bound project's region-clustering state dir: the job files and
    TTL markers below are per project (one project's refine must never
    suppress another's re-partition, nor its job show on another's status)."""
    from src.config.curation import get_curation_config

    return Path(get_curation_config().project_state_dir) / 'region_cluster'


def _job_file() -> Path:
    return _region_state_dir() / 'job.json'


_JOB_STALE_S = 1800.0  # a 'running' flag older than this is treated as dead
_DEFAULT_JOB: dict[str, Any] = {
    'running': False,
    'started_at': None,
    'finished_at': None,
    'result': None,
    'error': None,
}
_job_tasks: set[asyncio.Task[None]] = set()


def _read_region_cluster_job() -> dict[str, Any]:
    try:
        state: dict[str, Any] = json.loads(_job_file().read_text())
    except Exception:
        return dict(_DEFAULT_JOB)
    # Stale-guard: a worker that died mid-run would otherwise leave the flag
    # stuck on 'running' forever, wedging the button.
    if state.get('running') and state.get('started_at'):
        try:
            started = datetime.fromisoformat(state['started_at'])
            if (datetime.now(UTC) - started).total_seconds() > _JOB_STALE_S:
                state['running'] = False
                state['error'] = 'job timed out or worker died'
        except ValueError:
            pass
    return state


def _write_region_cluster_job(state: dict[str, Any]) -> None:
    try:
        _job_file().parent.mkdir(parents=True, exist_ok=True)
        tmp = _job_file().with_suffix('.tmp')
        tmp.write_text(json.dumps(state))
        tmp.replace(_job_file())  # atomic rename
    except Exception as exc:
        logger.warning('curation_region_cluster_job_write_failed', error=str(exc))


def region_cluster_job_status() -> dict[str, Any]:
    """Cross-worker snapshot of the background region-clustering job."""
    return _read_region_cluster_job()


# A manual AHC refine of a good region bucket writes per-crop sub-ids that a
# full re-partition would wipe (sub-ids are cluster-local). We record the last
# refine time so the one-click pipeline can skip the destructive re-partition
# while recent refine work is still fresh (a SHORT TTL), unless the caller forces
# it OR a substantial batch of new FPs has accumulated since the last partition
# (which busts the TTL — the good-region pool changed enough to be worth it).
REGION_REPARTITION_REFINE_TTL_S = 600.0  # 10 min — short; just protects in-progress refines
FP_REPARTITION_BUST_DELTA = 200  # this many new FPs since last partition busts the TTL


def _refine_marker() -> Path:
    return _region_state_dir() / 'refine_marker.json'


def _partition_marker() -> Path:
    return _region_state_dir() / 'partition_marker.json'


def mark_region_refine(cluster_id: int) -> None:
    """Record that a good region bucket was just manually refined (TTL anchor)."""
    try:
        _refine_marker().parent.mkdir(parents=True, exist_ok=True)
        tmp = _refine_marker().with_suffix('.tmp')
        tmp.write_text(
            json.dumps({'last_refine_at': datetime.now(UTC).isoformat(), 'cluster_id': cluster_id})
        )
        tmp.replace(_refine_marker())
    except Exception as exc:
        logger.warning('curation_region_refine_marker_write_failed', error=str(exc))


def _read_region_refine_marker() -> dict[str, Any]:
    try:
        data: dict[str, Any] = json.loads(_refine_marker().read_text())
        return data
    except Exception:
        return {}


def _write_region_partition_marker(fp_count: int) -> None:
    try:
        _partition_marker().parent.mkdir(parents=True, exist_ok=True)
        tmp = _partition_marker().with_suffix('.tmp')
        tmp.write_text(
            json.dumps({'last_partition_at': datetime.now(UTC).isoformat(), 'fp_count': fp_count})
        )
        tmp.replace(_partition_marker())
    except Exception as exc:
        logger.warning('curation_region_partition_marker_write_failed', error=str(exc))


def _read_region_partition_marker() -> dict[str, Any]:
    try:
        data: dict[str, Any] = json.loads(_partition_marker().read_text())
        return data
    except Exception:
        return {}


async def start_region_cluster_job(
    client: AsyncOpenSearch,
    *,
    max_rank: int | None = None,
    auto_fp_threshold: float | None = 0.20,
    rebuild_fp_centroids: bool = True,
    repartition_ttl_s: float = REGION_REPARTITION_REFINE_TTL_S,
    fp_bust_delta: int = FP_REPARTITION_BUST_DELTA,
    force_repartition: bool = False,
) -> dict[str, Any]:
    """Launch the full region-clustering pipeline in the background (single-flight).

    Pipeline (FP-prep steps are best-effort so a failure can't block the main
    re-partition):
      1. ``rebuild_fp_centroids``: re-sub-type the FP bucket + rebuild its
         sub-type centroids from the current false positives.
      2. ``auto_fp_threshold`` > 0: auto-move regions within that L2 distance of an
         FP sub-centroid into the FP bucket (the tight, near-certain matches).
      3. re-partition the good regions (FPs — including the just-moved ones —
         excluded), so the good buckets' centroids stay clean. **Skipped** when a
         manual region refine happened within ``repartition_ttl_s`` (a short TTL),
         to avoid wiping that fresh cluster-local sub-id work — UNLESS
         ``force_repartition`` or at least ``fp_bust_delta`` new FPs have
         accumulated since the last partition (a substantial change busts the TTL).

    Returns the job snapshot immediately so the caller never blocks. If a run is
    already in flight, returns its snapshot without starting another.
    """
    state = _read_region_cluster_job()
    if state.get('running'):
        return state
    started = datetime.now(UTC).isoformat()
    _write_region_cluster_job(
        {'running': True, 'started_at': started, 'finished_at': None, 'result': None, 'error': None}
    )

    async def _run() -> None:
        result: dict[str, Any] | None = None
        error: str | None = None
        extra: dict[str, Any] = {}
        try:
            if rebuild_fp_centroids:
                try:
                    extra['fp_centroids'] = await build_region_fp_centroids(client)
                except Exception as exc:
                    extra['fp_centroids'] = {'status': 'error', 'error': str(exc)}
                    logger.error('curation_region_job_fp_build_failed', error=str(exc))
            if auto_fp_threshold and auto_fp_threshold > 0:
                try:
                    extra['auto_fp'] = await auto_assign_fp_from_centroids(
                        client, threshold=auto_fp_threshold
                    )
                except Exception as exc:
                    extra['auto_fp'] = {'status': 'error', 'error': str(exc)}
                    logger.error('curation_region_job_auto_fp_failed', error=str(exc))
            # TTL gate: a re-partition clears good-region sub-ids, so skip it
            # while a recent manual refine is still fresh — UNLESS forced, or a
            # substantial batch of FPs accumulated since the last partition
            # (then the good-region pool changed enough to be worth re-clustering).
            current_fp = await count_false_positive_boxes(client)
            fp_at_last = int(_read_region_partition_marker().get('fp_count', 0))
            fp_delta = current_fp - fp_at_last
            refine_at = _read_region_refine_marker().get('last_refine_at')
            refine_fresh = False
            if refine_at and not force_repartition:
                try:
                    age = (datetime.now(UTC) - datetime.fromisoformat(refine_at)).total_seconds()
                    refine_fresh = age < repartition_ttl_s
                except ValueError:
                    refine_fresh = False
            busts_ttl = fp_delta >= fp_bust_delta
            do_repartition = force_repartition or busts_ttl or not refine_fresh
            if do_repartition:
                result = await cluster_region_residuals(client, max_rank=max_rank)
                _write_region_partition_marker(current_fp)
            else:
                result = {
                    'status': 'skipped_repartition_ttl',
                    'reason': 'recent manual region refine within TTL; sub-clusters preserved',
                    'last_refine_at': refine_at,
                    'repartition_ttl_s': repartition_ttl_s,
                    'fp_delta_since_partition': fp_delta,
                    'fp_bust_delta': fp_bust_delta,
                }
            result = {**result, **extra}
        except Exception as exc:
            error = str(exc)
            logger.error('curation_region_cluster_job_failed', error=str(exc))
        finally:
            _write_region_cluster_job(
                {
                    'running': False,
                    'started_at': started,
                    'finished_at': datetime.now(UTC).isoformat(),
                    'result': result,
                    'error': error,
                }
            )

    task = asyncio.create_task(_run())
    _job_tasks.add(task)
    task.add_done_callback(_job_tasks.discard)
    return _read_region_cluster_job()


# Background FP-centroid job state — same cross-worker file pattern as the
# region-clustering job above (own file so the two can run independently).
def _fp_job_file() -> Path:
    return _region_state_dir() / 'fp_job.json'


def _read_region_fp_job() -> dict[str, Any]:
    try:
        state: dict[str, Any] = json.loads(_fp_job_file().read_text())
    except Exception:
        return dict(_DEFAULT_JOB)
    if state.get('running') and state.get('started_at'):
        try:
            started = datetime.fromisoformat(state['started_at'])
            if (datetime.now(UTC) - started).total_seconds() > _JOB_STALE_S:
                state['running'] = False
                state['error'] = 'job timed out or worker died'
        except ValueError:
            pass
    return state


def _write_region_fp_job(state: dict[str, Any]) -> None:
    try:
        _fp_job_file().parent.mkdir(parents=True, exist_ok=True)
        tmp = _fp_job_file().with_suffix('.tmp')
        tmp.write_text(json.dumps(state))
        tmp.replace(_fp_job_file())
    except Exception as exc:
        logger.warning('curation_region_fp_job_write_failed', error=str(exc))


def region_fp_centroid_job_status() -> dict[str, Any]:
    """Cross-worker snapshot of the background FP-centroid build job.

    Merges in the persisted centroid metadata (``trained_at``/``k``/``n_boxes``)
    so the UI can warn when the centroids are stale.
    """
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    state = _read_region_fp_job()
    store = FalsePositiveCentroidStore()
    if store.load():
        state['centroids'] = {
            'trained_at': store.metadata.get('trained_at'),
            'k': store.metadata.get('k'),
            'n_boxes': store.metadata.get('n_boxes'),
        }
    else:
        state['centroids'] = None
    return state


async def start_region_fp_centroid_job(client: AsyncOpenSearch) -> dict[str, Any]:
    """Launch :func:`build_region_fp_centroids` in the background (single-flight)."""
    state = _read_region_fp_job()
    if state.get('running'):
        return state
    started = datetime.now(UTC).isoformat()
    _write_region_fp_job(
        {'running': True, 'started_at': started, 'finished_at': None, 'result': None, 'error': None}
    )

    async def _run() -> None:
        result: dict[str, Any] | None = None
        error: str | None = None
        try:
            result = await build_region_fp_centroids(client)
        except Exception as exc:
            error = str(exc)
            logger.error('curation_region_fp_job_failed', error=str(exc))
        finally:
            _write_region_fp_job(
                {
                    'running': False,
                    'started_at': started,
                    'finished_at': datetime.now(UTC).isoformat(),
                    'result': result,
                    'error': error,
                }
            )

    task = asyncio.create_task(_run())
    _job_tasks.add(task)
    task.add_done_callback(_job_tasks.discard)
    return region_fp_centroid_job_status()


__all__ = [
    'FP_REPARTITION_BUST_DELTA',
    'REGION_REPARTITION_REFINE_TTL_S',
    'mark_region_refine',
    'region_cluster_job_status',
    'region_fp_centroid_job_status',
    'start_region_cluster_job',
    'start_region_fp_centroid_job',
]
