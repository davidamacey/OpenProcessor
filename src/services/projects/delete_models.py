"""Model-sharing checks and unload of the Triton models a deleted project owns.

Split out of ``delete.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.logging import get_logger
from src.services.training.model_classes import owned_model_names


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


logger = get_logger(__name__)


async def _shared_model_users(record: ProjectRecord) -> list[str]:
    """§5.5 in_use guard: which of this project's own promoted models
    have opted into cross-project sharing (``PUT /models/{name}/sharing``,
    ``promote.json.shared``)? A strict subset of :func:`owned_model_names` --
    used ONLY to decide the ``in_use`` refusal (bypassable with
    ``force=True``), never to decide whether unload runs (see
    :func:`_unload_owned_models`, which unloads every owned model,
    shared or not).

    A project with any shared model is refused unless ``force=True``:
    "opted into sharing" is itself evidence someone may depend on it.
    :func:`~src.services.config_store.project_usage.model_dependents` names the projects that actually depend on
    one today (their ACTIVE detection profile names the model).
    """
    from src.services.training.model_classes import is_model_shared

    return [name for name in owned_model_names(record.slug) if is_model_shared(name)]


async def _unload_owned_models(record: ProjectRecord) -> list[str]:
    """M5 step 4 / P3F pass-3 MA2: unload every one of this project's
    own promoted models -- private and shared alike (:func:`owned_model_names`,
    the full ownership enumeration; NOT :func:`_shared_model_users`,
    which only narrows the ``in_use`` refusal) -- from Triton and remove
    their model repo directories, reusing P2's own unload primitive
    (``src.services.training.triton_promote.unload_triton_model``, the
    same one ``DELETE /models/{name}`` calls) rather than reimplementing
    Triton model removal here. A project's own models are ALWAYS cleaned
    up on delete; ``force`` only bypasses the upfront ``in_use`` 409 for
    the shared subset, never whether this step runs.

    Mirrors :func:`_delete_indexes`: collects (never swallows) failures
    so the caller can leave the record retryable instead of tombstoning
    a project some of whose models are still live in Triton."""
    from src.services.training.promote_errors import ModelNotPromotedError
    from src.services.training.triton_promote import unload_triton_model

    failed: list[str] = []
    for name in owned_model_names(record.slug):
        try:
            await unload_triton_model(name)
        except ModelNotPromotedError:
            # Already gone (e.g. a prior finish attempt's unload
            # succeeded before a later step in that same run failed) --
            # idempotent no-op, not a failure to retry.
            continue
        except Exception as exc:
            logger.warning(
                'project_delete_model_unload_failed',
                project=record.slug,
                model=name,
                error=str(exc),
            )
            failed.append(name)
    return failed
