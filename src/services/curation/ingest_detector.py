"""The ingest detector a project runs: the deployment's primary profile, or the
project's own model from its ingest policy.

``effective_profile`` is the one place a policy's detector override is laid over
the env profile, so ingest, the detector info, class seeding and a dataset
import's propose mode all see the same detector for a project. An override's
classes are never registry ids (``assigns_class`` is forced off).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from src.clients.model_adapters import END2END_OUTPUTS
from src.services.curation.ingest_policy_store import get_ingest_policy


if TYPE_CHECKING:
    from src.config import DetectionProfile
    from src.services.curation.ingest_policy import DetectorOverride, IngestPolicy


def effective_profile(
    base: DetectionProfile, override: DetectorOverride | None
) -> DetectionProfile:
    """``base`` with the project's detector laid over it (``base`` itself when
    the project has none)."""
    if override is None:
        return base
    changes: dict[str, Any] = {
        'detector_model': override.model,
        'detector_version': override.version,
        'labels_path': override.labels_path,
        'assigns_class': False,
    }
    if override.input_size is not None:
        changes['input_size'] = override.input_size
    return dataclasses.replace(base, **changes)


async def project_ingest_profile(
    client: Any, base: DetectionProfile
) -> tuple[DetectionProfile, IngestPolicy]:
    """The bound project's effective ingest detector and its policy."""
    policy = await get_ingest_policy(client)
    return effective_profile(base, policy.detector), policy


async def detector_problems(pool: Any, override: DetectorOverride) -> list[str]:
    """Why ``override`` cannot serve ingest (empty when it can): the model is not
    ready on Triton, or lacks the end2end output tensors."""
    if not await pool.is_model_ready(override.model):
        return [f'{override.model!r} is not loaded and ready on Triton']
    try:
        outputs = set(await pool.get_model_output_names(override.model))
    except Exception as exc:
        return [f'could not read {override.model!r} metadata from Triton: {exc}']
    missing = [name for name in END2END_OUTPUTS if name not in outputs]
    if missing:
        return [f'{override.model!r} does not serve the end2end outputs {missing}']
    return []


__all__ = ['detector_problems', 'effective_profile', 'project_ingest_profile']
