"""Promoted models for the project-delete and stats tests, as the rows a promote leaves.

``stage`` names a model before its owner exists; ``seed`` inserts the rows once the project is
created (the models table references the project). A shared model is shared with ``default``.
"""

from __future__ import annotations

from src.db.repos import models as repo, projects as projects_repo
from src.db.session import async_session_factory, deployment_tx, project_tx
from src.services.training import model_registry


_STAGED: list[tuple[str, str, bool]] = []


def stage(name: str, *, project: str, shared: bool = False) -> None:
    _STAGED.append((name, project, shared))


async def seed() -> None:
    staged, _STAGED[:] = list(_STAGED), []
    for name, slug, shared in staged:
        async with async_session_factory()() as session, deployment_tx(session, 'test seed'):
            owner = await projects_repo.get_by_slug(session, slug)
            consumer = await projects_repo.get_by_slug(session, 'default') if shared else None
        assert owner is not None, slug
        async with async_session_factory()() as session, project_tx(session, owner.id):
            await repo.record_promotion(
                session,
                model=repo.NewModel(
                    project_id=owner.id,
                    name=name,
                    origin='trained',
                    family='yolo26',
                    task='detect',
                    labels=['car'],
                    artifact_storage_key=f'models/{name}/1/model.onnx',
                    created_by='test',
                ),
                triton_model_name=name,
                triton_version=1,
                class_remap=None,
                job_id=None,
                promoted_by='test',
            )
        if shared:
            assert consumer is not None, 'default'
            await model_registry.set_sharing(
                owner.id,
                name,
                shared=True,
                expected_revision=1,
                consumers=[consumer.id],
                requested_by='test',
            )
