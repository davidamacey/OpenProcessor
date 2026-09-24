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
    'Run the auto-promote stage (classifier + cluster-majority agreement). '
    'Defaults False: the rule had no classifier confidence floor and was '
    'auto-validating low-confidence classifier predictions into class clusters. '
    'Opt-in only after a confidence-gated rewrite.'
)

RUN_VLM_DESC = (
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

CLUSTER_ID_DESC = (
    'Scope the VLM labeling stage to one cluster: every unvalidated, non-holdout, '
    "non-excluded member is labeled (the global sweep's cost skips do not apply). "
    'POST /vlm/label_cluster/{cluster_id} starts exactly this run.'
)

# detection_profile is NOT a per-run option: region detection runs in the
# detection worker on OP_REGION_PROFILE and no auto-label stage uses it.
# The param stays declared (hidden) only so an old client still sending it
# gets a clear 422 instead of FastAPI silently ignoring an unknown param.
DETECTION_PROFILE_REJECTED = (
    'detection_profile is not a per-run auto_label option: no auto_label stage '
    'runs region detection, and the detection worker uses the process region '
    'profile (OP_REGION_PROFILE). Remove the parameter.'
)

PROMPT_PACK_DESC = (
    'Per-run VLM prompt_pack (an id from GET /methods axis=prompt_pack) used by '
    'the labeling stage. Overrides the settings default for this job only; never '
    'written to settings. Unset = settings default. Unknown id -> 422.'
)


def reject_detection_profile(detection_profile: Any) -> None:
    """422 for any explicit ``detection_profile`` value (see
    :data:`DETECTION_PROFILE_REJECTED`). Non-``str`` means omitted."""
    if isinstance(detection_profile, str):
        raise HTTPException(
            status_code=422,
            detail={'error': DETECTION_PROFILE_REJECTED, 'param': 'detection_profile'},
        )


async def resolve_run_prompt_pack(opensearch: Any, prompt_pack: Any) -> str | None:
    """Resolve the per-run ``prompt_pack`` id.

    Non-``str`` values (``None``, or an unfilled FastAPI ``Query`` default
    when the endpoint function is called directly) mean "omitted" and
    resolve to the settings-doc default. An unknown id is a 422 listing
    the valid ids — never a silent fallback.
    """
    try:
        return await resolve_strategy_selection(
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
