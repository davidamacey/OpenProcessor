"""In-memory stand-ins for the training run service, for tests that are not about the database.

The run service (``src.services.training.jobs``, ``job_manifest``, ``job_campaigns``) reads and
writes PostgreSQL; a router, promote or bake-off test that only needs "a run with this status /
spec / manifest" installs these instead. The database behaviour itself is tested in
``tests/training``.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import pytest


def install_run_fakes(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace the run service seams; the returned namespace holds what they answer and record."""
    from src.services.training import (
        job_campaigns,
        job_manifest,
        jobs as train_jobs,
        preflight_run,
        trainer_reachability,
    )

    fake = SimpleNamespace(
        started=[],
        campaigns=[],
        active=None,
        reachable=('ok', 'trainer available'),
        gpu_order=None,
        cancelled=[],
        cancel_found=True,
        statuses={},
        logs={},
        manifests={},
        specs={},
        artifacts={},
        campaign_cancelled=0,
    )

    async def start_run(_session: Any, spec: Any, **kwargs: Any) -> str:
        fake.started.append((spec, kwargs))
        return str(uuid.uuid4())

    async def write_campaign(_session: Any, campaign: Any) -> tuple[str, list[str]]:
        fake.campaigns.append(campaign)
        return 'camp-1', [str(uuid.uuid4()) for _ in campaign.runs]

    async def get_active_job() -> Any:
        return fake.active

    async def probe() -> tuple[str, str]:
        return fake.reachable

    async def gpu_order() -> Any:
        return fake.gpu_order

    async def cancel_run(job_id: str) -> bool:
        fake.cancelled.append(job_id)
        return fake.cancel_found

    async def read_status(job_id: str) -> Any:
        return fake.statuses.get(job_id)

    async def list_runs(limit: int = 50, offset: int = 0) -> list[Any]:
        return list(fake.statuses.values())[offset : offset + limit]

    async def count_runs() -> int:
        return len(fake.statuses)

    async def tail_run_log(job_id: str, lines: int = 200) -> list[str]:
        return fake.logs.get(job_id, [])[-lines:]

    async def cancel_campaign(_campaign_id: str) -> int:
        return fake.campaign_cancelled

    async def read_manifest(job_id: str) -> Any:
        return fake.manifests.get(job_id)

    async def read_job_spec(job_id: str) -> Any:
        return fake.specs.get(job_id)

    async def read_artifact(job_id: str, name: str) -> Any:
        train_jobs.parse_run_id(job_id)
        return fake.artifacts.get((job_id, name))

    async def stamp_manifest_promotion(job_id: str, **promoted: Any) -> bool:
        if job_id not in fake.manifests:
            return False
        fake.manifests[job_id] = {**fake.manifests[job_id], 'promoted_to': promoted}
        return True

    monkeypatch.setattr(train_jobs, 'read_status', read_status)
    monkeypatch.setattr(train_jobs, 'list_runs', list_runs)
    monkeypatch.setattr(train_jobs, 'count_runs', count_runs)
    monkeypatch.setattr(train_jobs, 'tail_run_log', tail_run_log)
    monkeypatch.setattr(job_campaigns, 'cancel_campaign', cancel_campaign)
    monkeypatch.setattr(job_manifest, 'read_manifest', read_manifest)
    monkeypatch.setattr(job_manifest, 'read_artifact', read_artifact)
    monkeypatch.setattr(job_manifest, 'read_job_spec', read_job_spec)
    monkeypatch.setattr(job_manifest, 'stamp_manifest_promotion', stamp_manifest_promotion)
    monkeypatch.setattr(train_jobs, 'start_run', start_run)
    monkeypatch.setattr(train_jobs, 'cancel_run', cancel_run)
    monkeypatch.setattr(job_campaigns, 'write_campaign', write_campaign)
    monkeypatch.setattr(train_jobs, 'get_active_job', get_active_job)
    monkeypatch.setattr(preflight_run, 'probe_trainer_reachable', probe)
    monkeypatch.setattr(preflight_run, 'read_trainer_gpu_order', gpu_order)
    monkeypatch.setattr(trainer_reachability, 'read_trainer_gpu_order', gpu_order)
    from src.routers.curation.train import catalog

    monkeypatch.setattr(catalog, 'read_trainer_gpu_order', gpu_order)
    return fake
