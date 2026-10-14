"""The project-bound inputs every open-vocabulary validation needs."""

from __future__ import annotations

from typing import Any

from src.services.curation.class_registry_names import registry_class_names


def validation_inputs() -> dict[str, Any]:
    """The registry's class names, the primary detector's class names and the
    segmenter probe: the one place they are assembled, so create, save, validate,
    clone and the activation gate cannot disagree."""
    from src.config.ingest_profiles import ingest_primary_profile
    from src.services.detection.segmenter_health import configured_segmenter_health
    from src.services.labeling.vlm_endpoints import vlm_configured
    from src.utils.class_names import get_class_names

    detector = ingest_primary_profile().detector_model
    return {
        'class_names': registry_class_names(),
        'detector_class_names': frozenset(get_class_names(detector).values())
        if detector
        else frozenset(),
        'segmenter_health': configured_segmenter_health,
        'vlm_configured': vlm_configured(),
    }
