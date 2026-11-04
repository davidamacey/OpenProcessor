"""Map a detector's output index to a registry class by NAME.

A detector's class index is local to that model; the class registry is append-only and gets gaps.
The only identity that crosses the boundary is the name: the model's own name for the index
(its labels), then the active registry class carrying that name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config.ingest_profiles import model_class_name
from src.core.logging import get_logger
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from src.config import DetectionProfile


logger = get_logger(__name__)


def registry_class_for(
    profile: DetectionProfile, model_index: int, by_name: dict[str, tuple[int, str]]
) -> tuple[int, str] | None:
    """The active registry class ``(class_id, class_name)`` named like the model's output
    ``model_index``, or ``None`` (logged) when the model has no name for it or the registry has
    no active class of that name. ``by_name`` is ``registry_name_index(registry)``."""
    name = model_class_name(profile, model_index)
    found = by_name.get(normalize_class_name(name)) if name else None
    if found is None:
        # The index is context for the operator, never an identity.
        logger.info(
            'detector_class_unmapped',
            detector=profile.detector_model or profile.name,
            model_name=name,
            model_index=model_index,
        )
    return found
