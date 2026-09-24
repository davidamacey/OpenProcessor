"""Served region-detection vocabulary — ``GET {prefix}/regions/vocabulary``.

W0 of ``docs/design/naming_sweep_plan.md`` (finding m9): after S3/S7 the
worker's detector/segmenter/VLM identifiers come entirely from deployment
config (``DetectionProfile``, ``OP_VLM_*``), so the frontend can no longer
hardcode a label/palette map keyed on private model ids
(``lpr_nanov11_640``, ``sam3``, ``gemma-4-e4b``, ...). This module is the
single catalog a client renders from instead — mirrors the
``class_sources.py`` / ``GET {prefix}/class_sources`` pattern.

Never hardcodes a model id: everything here is read from the active
``DetectionProfile`` / ingest profiles / ``OP_VLM_MODEL`` at call time.
"""

from __future__ import annotations

import os
from typing import Any

from src.config.ingest_profiles import ingest_primary_profile, ingest_secondary_profile
from src.config.region_source import CANDIDATE_SOURCES
from src.services.detection.profile_registry import get_active_region_profile


# ``role`` values servable on a ``detectors`` / ``chain_actors`` entry.
VOCABULARY_ROLES: tuple[str, ...] = (
    'detector',
    'segmenter',
    'ocr',
    'verifier',
    'human',
    'classifier',
    'proposal',
)

_CANDIDATE_SOURCE_LABELS: dict[str, tuple[str, str]] = {
    'segmenter': ('Segmenter', 'segmenter'),
    'segmenter_text_hint': ('Segmenter (text hint re-pass)', 'segmenter'),
    'detector': ('Detector', 'detector'),
    'detector_existing': ('Detector (existing candidate)', 'detector'),
}


def _entry(entry_id: str, label: str, role: str, *, filterable: bool) -> dict[str, Any]:
    return {'id': entry_id, 'label': label, 'role': role, 'filterable': filterable}


def _detectors(vlm_model: str) -> list[dict[str, Any]]:
    """Every identifier that can appear in ``region_detector``,
    ``region_verifier``, ``class_labeler`` or ``class_detector`` — with
    ``filterable=True`` reserved for the ``region_detector`` values only
    (see ``docs/design/naming_sweep_plan.md`` W0)."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(entry_id: str, label: str, role: str, *, filterable: bool) -> None:
        if not entry_id or entry_id in seen:
            return
        seen.add(entry_id)
        out.append(_entry(entry_id, label, role, filterable=filterable))

    profile = get_active_region_profile()
    if profile is not None:
        add(
            profile.detector_model,
            f'Detector ({profile.detector_model})',
            'detector',
            filterable=True,
        )
        add(
            profile.segmenter_name,
            f'Segmenter ({profile.segmenter_name})',
            'segmenter',
            filterable=True,
        )
        add(profile.human_detector_name, 'Human', 'human', filterable=True)
        # The OCR text-detector only *locates* text to seed a segmenter
        # sub-crop re-pass; it never sets the region bbox itself, so it
        # never appears in region_detector -- filterable=False.
        add(profile.ocr_det_model, 'OCR text hint', 'ocr', filterable=False)
    else:
        add('human', 'Human', 'human', filterable=True)

    if vlm_model:
        add(vlm_model, f'VLM ({vlm_model})', 'verifier', filterable=False)

    primary = ingest_primary_profile()
    p_model = primary.detector_model or primary.name
    add(p_model, f'Proposer ({p_model})', 'proposal', filterable=False)

    secondary = ingest_secondary_profile()
    if secondary is not None:
        s_model = secondary.detector_model or secondary.name
        add(s_model, f'Classifier ({s_model})', 'classifier', filterable=False)

    return out


def _region_sources() -> list[dict[str, Any]]:
    """Every S3 ``region_source`` / ``candidate_source`` value, plus the
    fixed ``human`` value (a human-drawn/edited box)."""
    out: list[dict[str, Any]] = []
    for source_id in CANDIDATE_SOURCES:
        label, role = _CANDIDATE_SOURCE_LABELS[source_id]
        out.append({'id': source_id, 'label': label, 'role': role})
    out.append({'id': 'human', 'label': 'Human', 'role': 'human'})
    return out


def region_vocabulary_catalog() -> dict[str, Any]:
    """``{detectors, region_sources, chain_actors}`` for the active
    deployment config. Never hardcodes a private model id."""
    vlm_model = os.environ.get('OP_VLM_MODEL', '')
    detectors = _detectors(vlm_model)
    region_sources = _region_sources()
    # chain_actors: the identifiers that can appear in a
    # detector_chain tag (``f'{actor}:hit'`` etc). Same underlying
    # identifiers as detectors, filterable dropped (chain tags aren't a
    # region_detector filter value).
    chain_actors = [{'id': d['id'], 'label': d['label'], 'role': d['role']} for d in detectors]
    return {
        'detectors': detectors,
        'region_sources': region_sources,
        'chain_actors': chain_actors,
    }


__all__ = ['VOCABULARY_ROLES', 'region_vocabulary_catalog']
