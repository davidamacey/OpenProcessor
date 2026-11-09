"""P2: promoted Triton model names are namespaced per project
(projects_plan.md §5.3).

Covers: a non-default project's promote name is ``{prefix}{requested}``
(``model_prefix`` from ``resources_for_new`` -- ``'{slug}__'``);
the live promotion records the promoting project; a requested name
containing ``'__'`` is rejected before it can spoof the separator; and a
project's ``/models/status``-equivalent ownership check never lets one
project see or unload another's promoted model.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest


if TYPE_CHECKING:
    from pathlib import Path

from curation._promote_start_fake import no_database
from curation.triton_promote._helpers import FakeRecorder, write_float_image_onnx

from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.training.job_models import TrainJobStatus
from src.services.training.triton_promote import TritonPromoter


pytestmark = pytest.mark.unbound


def _record(slug: str) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base_curation_config()),
    )


@pytest.fixture
def fake_checkpoint(tmp_path: Path) -> Path:
    run_dir = tmp_path / 'jobs' / 'job-abc'
    run_dir.mkdir(parents=True)
    (run_dir / 'best.pt').write_bytes(b'not-a-real-checkpoint')
    write_float_image_onnx(run_dir / 'best.onnx')
    return run_dir / 'best.pt'


@pytest.fixture
def fake_status(fake_checkpoint: Path) -> TrainJobStatus:
    return TrainJobStatus(
        job_id='job-abc',
        state='finished',
        checkpoint_path=str(fake_checkpoint),
        eval={'map50': 0.9, 'per_class': []},
    )


@pytest.fixture
def scratch_models_dir(tmp_path: Path) -> Path:
    d = tmp_path / 'scratch_triton_models'
    d.mkdir()
    return d


def test_model_prefix_matches_resources_for_new_convention() -> None:
    """The exact convention `resources_for_new` already established --
    the router must use it, not invent a different one."""
    beta = resources_for_new('beta', base_curation_config())
    assert beta.model_prefix == 'beta__'
    default_prefix = resources_for_new('default', base_curation_config()).model_prefix
    # §5.3/§5.5: model_prefix is the one deliberate exception to D-A's
    # "no default special case" -- default stays unprefixed so every
    # pre-projects / core-pipeline model (never namespaced) keeps
    # resolving as default's own.
    assert default_prefix == ''


@pytest.mark.asyncio
async def test_promote_records_the_namespaced_name(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TritonPromoter, '_trigger_load', AsyncMock(return_value=True))
    promoter = TritonPromoter(triton_models_dir=scratch_models_dir, triton_http_url='http://unused')

    with bind_project(_record('beta')):
        from src.config.curation_accessors import get_curation_config

        cfg = get_curation_config()
        requested = 'detector_v3'
        triton_name = f'{cfg.model_prefix}{requested}'
        assert triton_name == 'beta__detector_v3'

        recorder = FakeRecorder({0: 'car'})
        result = await promoter.promote(
            status=fake_status,
            triton_name=triton_name,
            recorder=recorder,
        )

    assert result.triton_name == 'beta__detector_v3'
    assert recorder.recorded == [('beta__detector_v3', 1)]


@pytest.mark.asyncio
async def test_default_promote_name_is_unprefixed(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """default's empty model_prefix means its promoted names are exactly
    what the caller requested -- no behavior change for the live
    single-project stack."""
    monkeypatch.setattr(TritonPromoter, '_trigger_load', AsyncMock(return_value=True))
    promoter = TritonPromoter(triton_models_dir=scratch_models_dir, triton_http_url='http://unused')

    from src.config.curation_accessors import base_curation_config
    from src.config.projects import new_project_record

    with bind_project(new_project_record('default', base_curation_config())):
        from src.config.curation_accessors import get_curation_config

        cfg = get_curation_config()
        assert cfg.model_prefix == ''
        triton_name = f'{cfg.model_prefix}detector_v3'

    with bind_project(new_project_record('default', base_curation_config())):
        result = await promoter.promote(
            status=fake_status, triton_name=triton_name, recorder=FakeRecorder({0: 'car'})
        )
    assert result.triton_name == 'detector_v3'


def test_requested_name_with_reserved_separator_is_rejected(
    tmp_path, monkeypatch, not_stale_registry
) -> None:
    """The router-level guard (curation_train.promote_run) 422s before
    even looking up the job -- a requested name containing '__' would
    spoof the namespacing separator once prefixed. POST is a write, so
    M2's read-only gate needs a registry that actually refreshes (the
    process-wide default test registry is permanently stale by
    design), or it 409s before the router's own validation runs."""
    del not_stale_registry
    no_database(monkeypatch)

    from _curation_app import SCOPED, curation_test_app
    from fastapi.testclient import TestClient

    from src.routers.curation._common import _raw_opensearch_dep
    from src.routers.curation.train import router as curation_train_router

    app = curation_test_app(curation_train_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: AsyncMock()  # noqa: PLW0108  # FastAPI parses a class here as a dependency with parameters

    with TestClient(app) as client:
        resp = client.post(
            f'{SCOPED}/train/promote/nonexistent-job',
            json={'triton_name': 'alpha__x'},
        )
    assert resp.status_code == 422
    assert resp.json()['detail']['code'] == 'triton_name_reserved_separator'


@pytest.mark.asyncio
async def test_project_owns_model_prefix_isolation() -> None:
    """A project only owns model names under its own prefix; default
    (empty prefix) owns everything not claimed by another known
    project's prefix."""
    from src.services.projects.registry import ProjectRegistry, set_project_registry
    from src.services.training.promoted_models import project_owns_model

    beta = _record('beta')
    registry = ProjectRegistry()
    registry._by_slug = {'beta': beta}
    registry._revision = 0
    set_project_registry(registry)
    try:
        with bind_project(beta):
            assert await project_owns_model('beta__x') is True
            assert await project_owns_model('other__x') is False
            assert await project_owns_model('yolov11_small_trt_end2end') is False

        from src.config.curation_accessors import base_curation_config
        from src.config.projects import new_project_record

        with bind_project(new_project_record('default', base_curation_config())):
            assert await project_owns_model('yolov11_small_trt_end2end') is True
            assert await project_owns_model('beta__x') is False
    finally:
        set_project_registry(None)
