"""Proves the FastAPI startup lifespan actually invokes orphaned-job
reconciliation for all five affected modules plus the model-export task
directory (persistence-hardening Gap 1 + Gap 2; probe_job added in the
2026-09-25 multi-worker-state fix).

Monkeypatches each module's ``reconcile_orphaned_jobs`` /
``reconcile_orphaned_export_tasks`` before constructing the app so this
never touches a real job state dir or export directory; it only proves
the lifespan wiring calls them, not their internal correctness (covered
by ``tests/curation/test_job_reconcile.py`` and
``tests/test_model_export_persistence.py``).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


def test_lifespan_calls_reconcile_for_all_job_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services import model_export
    from src.services.curation import embedding_viz, probe_job
    from src.services.curation.autolabel import job as autolabel_job
    from src.services.curation.item_scores import job as item_scores_job
    from src.services.curation.selection import job as selection_job

    calls: list[str] = []

    def _tracker(name: str):
        def _inner(*_args, **_kwargs):
            calls.append(name)
            return False

        return _inner

    monkeypatch.setattr(item_scores_job, 'reconcile_orphaned_jobs', _tracker('item_scores'))
    monkeypatch.setattr(selection_job, 'reconcile_orphaned_jobs', _tracker('selection'))
    monkeypatch.setattr(embedding_viz, 'reconcile_orphaned_jobs', _tracker('embedding_viz'))
    monkeypatch.setattr(autolabel_job, 'reconcile_orphaned_jobs', _tracker('autolabel'))
    monkeypatch.setattr(probe_job, 'reconcile_orphaned_jobs', _tracker('probe'))
    monkeypatch.setattr(model_export, 'reconcile_orphaned_export_tasks', _tracker('model_export'))

    from src.main import app

    with TestClient(app):
        pass

    assert set(calls) == {
        'item_scores',
        'selection',
        'embedding_viz',
        'autolabel',
        'probe',
        'model_export',
    }


def test_lifespan_survives_a_module_raising_during_reconcile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One misbehaving module must not block startup or the other three
    reconciliation calls -- each is wrapped independently in main.py."""
    from src.services.curation import embedding_viz
    from src.services.curation.item_scores import job as item_scores_job

    calls: list[str] = []

    def _boom(*_args, **_kwargs):
        raise RuntimeError('simulated reconcile failure')

    def _tracker(*_args, **_kwargs):
        calls.append('embedding_viz')
        return False

    monkeypatch.setattr(item_scores_job, 'reconcile_orphaned_jobs', _boom)
    monkeypatch.setattr(embedding_viz, 'reconcile_orphaned_jobs', _tracker)

    from src.main import app

    with TestClient(app):
        pass

    assert calls == ['embedding_viz']
