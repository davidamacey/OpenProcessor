"""Soft-delete a deleted project's MLflow experiment over the tracking
server's REST API. The API image carries no ``mlflow`` package, so this
speaks HTTP (``MLFLOW_TRACKING_URI``) instead of importing it."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx


_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class MlflowCleanup:
    """Outcome of the experiment cleanup, reported on ``project.deleted``."""

    outcome: Literal['done', 'skipped_not_configured', 'failed']
    reason: str | None = None

    def to_wire(self) -> dict[str, str | None]:
        return {'outcome': self.outcome, 'reason': self.reason}


async def delete_experiment(
    experiment_name: str, *, http: httpx.AsyncClient | None = None
) -> MlflowCleanup:
    """Soft-delete ``experiment_name`` (restorable on the server). Never
    raises: an unconfigured or unreachable server is an outcome, not a
    delete failure. A missing experiment counts as done."""
    base = (os.environ.get('MLFLOW_TRACKING_URI') or '').strip().rstrip('/')
    if not base:
        return MlflowCleanup('skipped_not_configured', 'MLFLOW_TRACKING_URI is not set')
    owned = http is None
    client: Any = http or httpx.AsyncClient(timeout=_TIMEOUT_S)
    try:
        found = await client.get(
            f'{base}/api/2.0/mlflow/experiments/get-by-name',
            params={'experiment_name': experiment_name},
        )
        if found.status_code == 404:
            return MlflowCleanup('done', 'experiment not found')
        found.raise_for_status()
        experiment_id = (found.json().get('experiment') or {}).get('experiment_id')
        if not experiment_id:
            return MlflowCleanup('failed', 'tracking server returned no experiment_id')
        deleted = await client.post(
            f'{base}/api/2.0/mlflow/experiments/delete', json={'experiment_id': experiment_id}
        )
        deleted.raise_for_status()
        return MlflowCleanup('done')
    except Exception as exc:
        return MlflowCleanup('failed', f'{type(exc).__name__}: {exc}')
    finally:
        if owned:
            await client.aclose()
