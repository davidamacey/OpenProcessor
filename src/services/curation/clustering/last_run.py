"""Durable record of the most recent residual-clustering run.

The dashboard's "Last clustering" card must reflect every run, whichever
path started it: the auto-label job worker, the synchronous
``POST /pipeline/auto_label`` the cluster-refresh daemon calls (which runs
in the API process and writes no job state), or a direct call. The one
code path they all share is :func:`orchestrator.cluster_residuals` /
:func:`orchestrator.assign_only_residuals`, so those record here. The file
lives in the bound project's ``autolabel_dir``, the directory the API and
the workers already share.
"""

from __future__ import annotations

import contextlib
import json
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.config.curation import get_curation_config


if TYPE_CHECKING:
    from pathlib import Path


_FILE_NAME = 'last_clustering.json'


def _path() -> Path:
    return get_curation_config().autolabel_dir / _FILE_NAME


def record_last_run(result: dict[str, Any]) -> None:
    """Persist a successful run's summary. Best effort: a write failure must
    not fail the clustering run that already changed the index."""
    record = {
        'finished_at': datetime.now(UTC).isoformat(),
        'method': result.get('method'),
        'n_clusters': result.get('n_clusters'),
        'n_residuals': result.get('n_residuals'),
        'n_noise': result.get('n_noise'),
    }
    path = _path()
    tmp = path.with_name(f'{_FILE_NAME}.{os.getpid()}.tmp')
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(record))
        tmp.replace(path)


def read_last_run() -> dict[str, Any] | None:
    """The recorded run, or ``None`` when none was recorded or it is unreadable."""
    try:
        data = json.loads(_path().read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
