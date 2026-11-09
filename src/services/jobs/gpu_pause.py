"""Whether the GPU is claimed, read from ``deployment_settings('vlm_paused_for_training')``.

The document is written by the arbiter tick and by ``openprocessor vlm use``
(:mod:`src.domain.gpu_arbiter_state`); this is the read side for the API.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from src.db.models import DeploymentSetting
from src.db.session import async_session_factory, deployment_tx
from src.domain import gpu_arbiter_state as state


async def _document() -> object:
    async with async_session_factory()() as session, deployment_tx(session, 'gpu pause state'):
        return (
            await session.execute(
                select(DeploymentSetting.value).where(DeploymentSetting.key == state.KEY)
            )
        ).scalar_one_or_none()


async def training_claim_reason() -> str | None:
    """Why every project's workers idle on a GPU claim, ``None`` when no run holds cards."""
    claims = state.arbiter_record(await _document()).claims
    if not claims:
        return None
    return f'GPU claim active ({len(claims)} running job{"s" if len(claims) != 1 else ""})'


async def vlm_paused() -> bool:
    """Whether the VLM workers must idle (a GPU claim or a model swap).

    A database that cannot be read counts as paused: a worker that cannot see the claim must not
    race a run for the card, and it cannot read its projects either.
    """
    try:
        return state.is_paused(await _document())
    except SQLAlchemyError:
        return True
