"""Proves the FastAPI startup lifespan actually invokes orphaned-job
reconciliation for the modules still file-backed (persistence-hardening Gap 1). Model-export
tasks are not reconciled here: the model-export worker owns them
(``tests/test_model_export_worker.py``).

Monkeypatches each module's ``reconcile_orphaned_jobs`` before constructing
the app so this never touches a real job state dir; it only proves the
lifespan wiring calls them, not their internal correctness (covered by
``tests/curation/test_job_reconcile.py``).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


def test_lifespan_calls_reconcile_for_all_job_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation.autolabel import job as autolabel_job

    calls: list[str] = []

    def _tracker(name: str):
        def _inner(*_args, **_kwargs):
            calls.append(name)
            return False

        return _inner

    monkeypatch.setattr(autolabel_job, 'reconcile_orphaned_jobs', _tracker('autolabel'))

    from src.main import app

    with TestClient(app):
        pass

    assert set(calls) == {'autolabel'}


def test_lifespan_survives_a_module_raising_during_reconcile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One misbehaving module must not block startup or the other three
    reconciliation calls -- each is wrapped independently in src/app/lifespan.py."""
    from src.services.curation.autolabel import job as autolabel_job

    calls: list[str] = []

    def _boom(*_args, **_kwargs):
        raise RuntimeError('simulated reconcile failure')

    monkeypatch.setattr(autolabel_job, 'reconcile_orphaned_jobs', _boom)

    from src.main import app

    with TestClient(app):
        pass

    assert calls == []  # startup completed although the module raised
