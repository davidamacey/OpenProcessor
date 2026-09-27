"""Served region-detection vocabulary — ``GET {prefix}/regions/vocabulary``.

After the naming sweep, the
worker's detector/segmenter/VLM identifiers come entirely from deployment
config (``DetectionProfile``, ``OP_VLM_*``), so the frontend can no longer
hardcode a label/palette map keyed on deployment model ids
(``my_region_det_640``, ``sam3``, ...). This module is the
single catalog a client renders from instead — mirrors the
``class_sources.py`` / ``GET {prefix}/class_sources`` pattern.

Never hardcodes a model id: everything here is read from the active
``DetectionProfile`` / ingest profiles / ``OP_VLM_MODEL`` at call time.
"""

from __future__ import annotations

import os
from typing import Any

from src.config.ingest_profiles import ingest_primary_profile, ingest_secondary_profile
from src.config.region_rejection import rejection_reason_catalog
from src.config.region_source import CANDIDATE_SEGMENTER_TEXT_HINT, CANDIDATE_SOURCES
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
    (see this module's docstring)."""
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
        if _text_hint_on(profile):
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


def _text_hint_on(profile: Any) -> bool:
    # The segmenter URL is worker config the API can't see; a profile with
    # the hint on and an OCR pipeline is advertised as able to run it.
    return profile.text_hint_active(segmenter_enabled=True)


def _region_sources(profile: Any) -> list[dict[str, Any]]:
    """Every ``region_source`` / ``candidate_source`` value the profile can
    write, plus the fixed ``human`` value (a human-drawn/edited box)."""
    out: list[dict[str, Any]] = []
    for source_id in CANDIDATE_SOURCES:
        if source_id == CANDIDATE_SEGMENTER_TEXT_HINT and not _text_hint_on(profile):
            continue
        label, role = _CANDIDATE_SOURCE_LABELS[source_id]
        out.append({'id': source_id, 'label': label, 'role': role})
    out.append({'id': 'human', 'label': 'Human', 'role': 'human'})
    return out


def _text_rules(profile: Any) -> dict[str, Any] | None:
    """The profile's region-text validity rules (with the resolved prompt
    pack's example values as placeholders), or ``None`` for a text-free
    profile."""
    from src.services.detection.region_text_rules import region_text_rules

    return region_text_rules(profile).catalog() if profile.reads_text else None


def region_profile_summary(profile: Any) -> dict[str, Any]:
    """``{name, display_name, display_name_singular, region_class_name,
    text_reader, reads_text, text_hint_enabled}`` for one
    ``DetectionProfile`` -- served on ``GET /health`` and
    ``GET /regions/vocabulary``. THE signal a client keys on to decide
    whether region-scoped UI/routes are available; ``reads_text`` is the
    one to key text UI on (``text_reader`` is ``'none'`` when false)."""
    return {
        'name': profile.name,
        'display_name': profile.display_name,
        'display_name_singular': profile.display_name_singular,
        'region_class_name': profile.region_class_name,
        'text_reader': profile.text_reader,
        'reads_text': profile.reads_text,
        'text_hint_enabled': profile.text_hint_enabled,
    }


def region_vocabulary_catalog() -> dict[str, Any]:
    """``{region_profile, detectors, region_sources, chain_actors,
    text_rules, text_choices, rejection_reasons}`` for the active
    deployment config. Never hardcodes a private model id.

    No-profile gating contract: with no active region profile, every
    list is empty, ``text_rules`` and ``region_profile`` are ``None`` --
    this endpoint still 200s (never 404s); ``region_profile`` is the
    signal a client checks first. A text-free profile serves the same
    empty text vocabulary (``text_rules: None``, ``text_choices: []``).
    """
    profile = get_active_region_profile()
    if profile is None:
        return {
            'region_profile': None,
            'detectors': [],
            'region_sources': [],
            'chain_actors': [],
            'text_rules': None,
            'text_choices': [],
            'rejection_reasons': [],
        }

    from src.services.detection.region_text import TEXT_CHOICES

    vlm_model = os.environ.get('OP_VLM_MODEL', '')
    detectors = _detectors(vlm_model)
    region_sources = _region_sources(profile)
    # chain_actors: the identifiers that can appear in a
    # detector_chain tag (``f'{actor}:hit'`` etc). Same underlying
    # identifiers as detectors, filterable dropped (chain tags aren't a
    # region_detector filter value).
    chain_actors = [{'id': d['id'], 'label': d['label'], 'role': d['role']} for d in detectors]
    return {
        'region_profile': region_profile_summary(profile),
        'detectors': detectors,
        'region_sources': region_sources,
        'chain_actors': chain_actors,
        'text_rules': _text_rules(profile),
        'text_choices': list(TEXT_CHOICES) if profile.reads_text else [],
        'rejection_reasons': rejection_reason_catalog(),
    }


__all__ = ['VOCABULARY_ROLES', 'region_profile_summary', 'region_vocabulary_catalog']
