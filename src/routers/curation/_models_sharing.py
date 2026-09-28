"""``PUT /models/{model_name}/sharing`` -- opt-in cross-project model
sharing (projects_plan.md §5.5, owner D1).

Split out of ``models.py`` to stay under the repo's 700-LOC pre-commit
ratchet.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import Query
from pydantic import BaseModel

from src.config.curation import get_curation_config
from src.routers.curation._common import logger, router
from src.routers.curation._config_common_models import api_error
from src.services.training.promote_json import SharingRevisionConflictError, update_sharing


def _promote_json_path(model_name: str) -> Any:
    from src.services.training.triton_promote import resolve_triton_models_dir

    return resolve_triton_models_dir() / model_name / 'promote.json'


class ModelSharingRequest(BaseModel):
    shared: bool
    expected_revision: int


class ModelSharingUser(BaseModel):
    project: str
    profile: str | None = None


class ModelSharingResponse(BaseModel):
    name: str
    project: str
    shared: bool
    revision: int
    used_by: list[ModelSharingUser] = []


@router.put('/models/{model_name}/sharing', response_model=ModelSharingResponse)
async def set_model_sharing(
    model_name: str,
    payload: ModelSharingRequest,
    force: Annotated[
        bool,
        Query(description='Bypass the in-use refusal when unsharing (logged)'),
    ] = False,
) -> ModelSharingResponse:
    """Opt a promoted model into (or out of) cross-project sharing. Only
    the owning project may call this -- 404 for anyone else, matching
    every other ownership check in this router."""
    from src.services.training.promoted_models import project_owns_model

    project = get_curation_config().project_slug
    if not project_owns_model(model_name):
        raise api_error(
            404,
            'model_not_found',
            f'{model_name!r} is not a model owned by this project',
            project=project,
        )

    # Unsharing while another project's active detector profile still
    # names this model would silently break that project's pipeline.
    # TODO(W4/profile_validation, out of scope here): a real cross-project
    # "who has this as their active detector_model" scan needs each
    # project's own bound DetectionProfile read, which the profile-CRUD
    # wave (W4) owns. used_by is always [] until that lands; unsharing is
    # never refused here yet.
    used_by: list[ModelSharingUser] = []
    if not payload.shared and used_by and not force:
        raise api_error(
            409,
            'in_use',
            f'{model_name!r} is still used by {len(used_by)} other project(s)',
            projects=[u.project for u in used_by],
        )
    if not payload.shared and used_by and force:
        logger.warning(
            'model_unshared_in_use',
            model_name=model_name,
            projects=[u.project for u in used_by],
        )

    try:
        revision = await asyncio.to_thread(
            update_sharing,
            _promote_json_path(model_name),
            shared=payload.shared,
            expected_revision=payload.expected_revision,
        )
    except SharingRevisionConflictError as exc:
        raise api_error(
            409,
            'revision_conflict',
            f'{model_name!r} sharing revision is {exc.current_revision}, '
            f'not {payload.expected_revision}',
            current_revision=exc.current_revision,
        ) from exc
    except (OSError, ValueError) as exc:
        raise api_error(
            404,
            'model_not_found',
            f'{model_name!r} has no promote.json to share (not promoted through this pipeline)',
            project=project,
        ) from exc

    return ModelSharingResponse(
        name=model_name,
        project=project,
        shared=payload.shared,
        revision=revision,
        used_by=used_by,
    )


__all__ = ['ModelSharingRequest', 'ModelSharingResponse', 'ModelSharingUser', 'set_model_sharing']
