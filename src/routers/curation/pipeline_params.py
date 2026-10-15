"""Shared query-param descriptions for the auto-label endpoints in
the auto-label routes (split out to keep those modules under the
file-size ceiling). No routes live here."""

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

VLM_SCOPE_DESC = (
    "Override the project's VLM scope policy (GET /vlm/policy) for this run: all, uncertain, "
    'representatives or off. Unset uses the stored policy. Ignored with cluster_id, an '
    'explicit request that no policy limits.'
)

REASSIGN_ONLY_DESC = 'IVF: stream-assign residuals vs persisted centroids; skip retrain.'

# Labeling-assist item selection (task d): scope a run to one registry class.
CLASS_ID_DESC = (
    'Scope the VLM labeling stage to a single registry class (labeling-assist '
    'item selection); its cluster-id upkeep touches only the selected items. '
    'Clustering and auto-promote are not scoped. Unset runs '
    'the full unvalidated cohort, unchanged from before this parameter existed.'
)

CLUSTER_ID_DESC = (
    'Scope the run to one cluster: every unvalidated, non-holdout, non-excluded '
    "member is VLM-labeled (the global sweep's cost skips do not apply) and every "
    'write stays on those members, so the index-wide clustering and auto-promote '
    'stages are skipped. For a sampled, per-cluster or per-sub-cluster run use POST /assist/runs.'
)

PROMPT_PACK_DESC = (
    'Per-run VLM prompt_pack (an id from GET /methods axis=prompt_pack) used by '
    'the labeling stage. Overrides the settings default for this job only; never '
    'written to settings. Unset = settings default. Unknown id -> 422.'
)

VLM_DESC = (
    'Per-run VLM endpoint (a name, or name@revision, from GET /methods axis=vlm) used by the '
    "labeling stage. Overrides this project's default for this run only; never written to "
    'settings. Unset = the project default. Unknown -> 422. An endpoint outside this deployment '
    "also needs acknowledge_external=true, unless it is already this project's acknowledged "
    'default.'
)
ACKNOWLEDGE_EXTERNAL_DESC = (
    'Acknowledge that crops will be sent to a VLM endpoint outside this deployment for this run.'
)
