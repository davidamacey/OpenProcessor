"""Shared query-param descriptions and per-run strategy selection for the
auto-label endpoints in :mod:`src.routers.curation.pipeline` (split out to
keep that module under the file-size ceiling). No routes live here."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from src.services.curation.strategy_defaults import UnknownStrategyError, resolve_strategy_selection


# Shared description for the run_auto_promote query param so both
# /start and /pipeline_auto_label entry points say the same thing.
AUTO_PROMOTE_DESC = (
    'Run the auto-promote stage (v6 + cluster-majority agreement). '
    'Defaults False: the rule had no v6 confidence floor and was '
    'auto-validating low-confidence v6 predictions into class clusters. '
    'Opt-in only after a confidence-gated rewrite.'
)

RUN_GEMMA_DESC = (
    'Run the VLM labeling stage. Defaults to False — the '
    'detection worker now labels crops on the drain path; this '
    'stage just duplicates that work. Opt-in for a one-off '
    'no-region-cohort backfill.'
)

REASSIGN_ONLY_DESC = 'IVF: stream-assign residuals vs persisted centroids; skip retrain.'

# Labeling-assist item selection (task d): scope a run to one registry class.
CLASS_ID_DESC = (
    'Scope the VLM labeling stage to a single registry class (labeling-assist '
    'item selection). Clustering and auto-promote are not scoped. Unset runs '
    'the full unvalidated cohort, unchanged from before this parameter existed.'
)

DETECTION_PROFILE_DESC = (
    'Per-run detection_profile (an id from GET /methods axis=detection_profile). '
    'Overrides the settings default for this job only; never written to settings. '
    'Unset = settings default. Unknown id -> 422.'
)

PROMPT_PACK_DESC = (
    'Per-run VLM prompt_pack (an id from GET /methods axis=prompt_pack) used by '
    'the labeling stage. Overrides the settings default for this job only; never '
    'written to settings. Unset = settings default. Unknown id -> 422.'
)


async def resolve_run_selection(
    opensearch: Any, detection_profile: Any, prompt_pack: Any
) -> tuple[str | None, str | None]:
    """Resolve the per-run ``detection_profile`` / ``prompt_pack`` ids.

    Non-``str`` values (``None``, or an unfilled FastAPI ``Query`` default
    when the endpoint function is called directly) mean "omitted" and
    resolve to the settings-doc default. An unknown id is a 422 listing
    the valid ids — never a silent fallback.
    """
    try:
        profile = await resolve_strategy_selection(
            'detection_profile',
            detection_profile if isinstance(detection_profile, str) else None,
            opensearch,
        )
        pack = await resolve_strategy_selection(
            'prompt_pack', prompt_pack if isinstance(prompt_pack, str) else None, opensearch
        )
    except UnknownStrategyError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                'error': str(exc),
                'axis': exc.axis,
                'requested': exc.requested,
                'valid_ids': exc.valid,
            },
        ) from exc
    return profile, pack
