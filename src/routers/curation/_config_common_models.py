"""Shared structured-error and validation models for the config-store /
projects surface (P1 creates this module ahead of W2, because P1 lands
first in the merge order -- see
docs/design/openprocessor_internal/projects_plan.md §7 and
any_domain_plan.md §7.1/§9 W2). W2 extends the Literals and adds the
validation-report shapes and ``ActiveRef``/``ActiveConfigResponse``;
W3/W4/W8/W9 extend the Literals further as their routes land.

Every project or config route raises through :func:`api_error`, never a
bare string ``detail`` -- gives Cropwright one stable shape
(``{"detail": ConfigErrorDetail}``) to parse everywhere.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel

from src.routers.curation._dataset_issue_models import (
    DatasetIssueWire,  # noqa: TC001 - pydantic field type, resolved at runtime
)


# P1 seeded the project-related codes it raises. W2 adds the codes its
# own low-level primitives (``src.services.config_store``) and the
# ``PUT /settings`` bridge raise; W3/W4/W8/W9 add the rest as their
# routes land (see any_domain_plan.md §7.1's full table).
ErrorCode = Literal[
    'project_not_found',
    'project_archived',
    'project_read_only',
    'project_building',
    'project_deleting',
    'project_failed',
    'project_busy',
    'slug_taken',
    'slug_retired',
    'last_active_project',
    'shard_budget_exceeded',
    'project_protected',
    'target_not_empty',
    'preview_stale',
    'in_use',
    'slug_invalid',
    'confirm_mismatch',
    'combine_invalid',
    'clone_source_not_ready',
    'model_name_reserved',
    'internal_isolation_error',
    'revision_conflict',
    # W2
    'not_found',
    'unknown_revision',
    'active_conflict',
    'no_previous',
    # m-b fix (W3/W4 round-5 review): rollback's `previous` target was
    # deleted since it was activated -- distinct from `no_previous`
    # (there was never a target at all).
    'previous_deleted',
    'no_active_profile',
    'unknown_pack',
    'unknown_profile',
    'validation_failed',
    'config_store_unavailable',
    'read_only',
    'name_conflict',
    'invalid_transition',
    'export_outside_project',
    'model_not_found',
    # W2b
    'class_hotkey_conflict',
    'hotkey_reserved',
    'hotkey_taken',
    # P3F m1: a delete-path directory guard refused because the persisted
    # record's own path pointed outside its expected root -- distinct
    # from internal_isolation_error (an OpenSearch-guard refusal).
    'path_escape',
    # P3F pass-3 MA1: a second delete_project_finish for the same slug
    # was refused because a first finish for it is still in flight --
    # distinct from project_busy (a step *inside* one finish failed).
    'finish_in_progress',
    # W10: dataset import and reprocess.
    'dataset_not_found',
    'import_not_found',
    'upload_not_found',
    'image_not_found',
    'import_busy',
    'import_resumable',
    'dataset_changed',
    'import_not_resumable',
    'import_not_undoable',
    'reprocess_busy',
    'upload_too_large',
    'dataset_path_not_allowed',
    'format_undetected',
    'import_blocked',
    'class_mapping_incomplete',
    'class_mapping_invalid',
    'archive_invalid',
    'reprocess_targets_invalid',
    'region_profile_required',
    # P4: combine projects.
    'combine_not_found',
    'combine_not_resumable',
    # W9: VLM endpoint registry / selection
    'unknown_vlm',
    'vlm_external_not_acknowledged',
    'vlm_not_configured',
    'vlm_endpoint_unavailable',
    'no_local_vlm',
    'unknown_catalog_id',
    'vlm_catalog_does_not_fit',
    'probe_busy',
    # W5: the test-on-crop routes
    'crop_not_found',
    'test_busy',
    'test_timeout',
    'vlm_transport_error',
    'no_box_to_verify',
    'no_class_names',
    'too_many_crops',
    'too_many_crop_ids',
    'pack_invalid',
    'profile_invalid',
    'segmenter_error',
    'detector_error',
]

# Seeded with the codes W2 raises (none yet -- W2 has no validated
# writes of its own, only the low-level OCC primitives). W3/W4 add the
# pack/profile validation codes; W8/W9 add theirs. W2b adds the
# keymap_* codes (CW-K §3.1).
ValidationCode = Literal[
    'name_conflict',
    'keymap_unknown_action',
    'keymap_action_locked',
    'keymap_combo_invalid',
    'keymap_too_many_combos',
    'keymap_key_locked',
    'keymap_browser_reserved',
    'keymap_context_collision',
    'keymap_overlay_unbound',
    'keymap_class_hotkey_conflict',
    'keymap_class_hotkey_shadowed',
    'keymap_focus_key',
    'keymap_context_no_confirm',
    # W3: prompt-pack validation (any_domain_plan.md §3.3/§7.1)
    'pack_name_invalid',
    'pack_name_reserved',
    'pack_field_missing',
    'pack_field_empty',
    'pack_field_too_long',
    'pack_placeholder_missing',
    'pack_placeholder_unknown',
    'pack_template_format_error',
    'pack_placeholder_in_verbatim_field',
    'pack_reply_key_missing',
    'pack_asks_text_profile_text_free',
    'pack_no_text_profile_reads_text',
    'pack_synonym_target_unknown',
    'pack_description_class_unknown',
    'pack_example_values',
    'pack_multi_region_keys_missing',
    # W4: region-profile validation (any_domain_plan.md §4.3/§7.1)
    'profile_name_invalid',
    'profile_name_reserved',
    'profile_field_unknown',
    'profile_field_type',
    'profile_field_range',
    'region_class_name_invalid',
    'text_reader_invalid',
    'text_regex_invalid',
    'detector_model_not_found',
    'detector_model_not_ready',
    'detector_model_not_shared',
    'detector_model_other_project',
    'detector_model_classes_unmapped',
    'triton_unreachable',
    'no_candidate_source',
    'segmenter_prompt_empty',
    'segmenter_prompt_too_long',
    'segmenter_prompt_too_many_phrases',
    'segmenter_prompt_multiline',
    'segmenter_not_configured',
    'segmenter_unreachable',
    'ocr_model_not_found',
    'ocr_model_not_ready',
    'vlm_not_configured',
    'text_hint_inactive_no_ocr',
    'text_fields_ignored',
    'display_name_missing',
    'parent_class_unknown',
    # Open-vocabulary prompt sets
    'open_vocab_name_invalid',
    'open_vocab_name_reserved',
    'open_vocab_field_invalid',
    'open_vocab_field_range',
    'open_vocab_no_enabled_targets',
    'open_vocab_too_many_targets',
    'open_vocab_duplicate_target',
    'open_vocab_class_name_invalid',
    'open_vocab_detector_class',
    'open_vocab_class_new',
    'open_vocab_vlm_not_configured',
    # P4: combine-projects preview / start validation.
    'unmapped_class',
    'mapping_target_invalid',
    'source_not_found',
    'source_busy',
    'source_not_ready',
    'slug_taken',
    'slug_retired',
    'slug_invalid',
    'shard_budget_exceeded',
    'shard_budget_high',
    'too_many_sources',
    'duplicate_source',
    'target_is_source',
    'label_conflicts',
    'holdout_recompute_contamination',
    'embedding_model_mismatch',
    'region_profiles_differ',
    'class_mapping_invalid',
    # W9: VLM endpoint validation (any_domain_plan.md W9.4/W9.5)
    'vlm_name_invalid',
    'vlm_name_reserved',
    'vlm_field_range',
    'vlm_url_invalid',
    'vlm_url_denied_internal_service',
    'vlm_url_denied_address',
    'vlm_api_key_ref_invalid',
    'vlm_api_key_unresolved',
    'vlm_api_key_ref_dropped',
    'vlm_external_not_acknowledged',
    'vlm_external_denied',
    'vlm_catalog_id_unknown',
    'vlm_unreachable',
    'vlm_timeout',
    'vlm_auth_failed',
    'vlm_http_error',
    'vlm_models_endpoint_missing',
    'vlm_model_not_listed',
    'vlm_no_vision',
    'vlm_vision_answer_wrong',
    'vlm_json_mode_unsupported',
    'vlm_max_images_exceeds_server',
    'vlm_reply_unparseable',
    'vlm_not_probed',
    'vlm_probe_failed',
    'vlm_context_too_small',
    'vlm_multi_box_unverified',
    'vlm_reads_text_unverified',
    'vlm_json_mode_off',
    'vlm_open_images_clamped',
]


class ProjectCapacityWire(BaseModel):
    """The OpenSearch shard/heap capacity block (§2.3,
    ``src.services.projects.capacity.ProjectCapacity.to_wire``): served on
    ``GET /projects`` and on a 409 ``shard_budget_exceeded``."""

    status: Literal['ok', 'warn', 'blocked']
    active_shards: int
    per_project_shards: int
    soft_limit: int
    hard_limit: int
    heap_max_bytes: int
    max_shards_per_node: int
    data_nodes: int
    projects_until_soft_limit: int
    message: str
    labels: dict[str, str]


class JobRefWire(BaseModel):
    """One running job blocking a lifecycle action (Cropwright rev-3
    delta 11): typed, not a raw id string, so a caller can render
    "cars has 2 running jobs: Training run run-7, Bake-off run-42"
    without a second lookup."""

    kind: str
    kind_label: str
    id: str
    label: str
    # P3F m5: a real timestamp when the job source has one, else null --
    # never the empty-string filler this used to always carry.
    started_at: str | None = None


class ModelSharingUser(BaseModel):
    """One other project whose active detection profile uses a shared model."""

    project: str
    profile: str | None = None


class ConfigErrorDetail(BaseModel):
    """The ``detail`` body of every project/config-store route's 4xx/5xx.

    Optional fields are code-specific and ``None``/absent otherwise; kept
    typed so OpenAPI documents them rather than leaving ``detail`` as an
    opaque blob.
    """

    error: ErrorCode
    message: str
    project: str | None = None
    # delta 11: 409 project_busy carries typed JobRef objects, not raw ids.
    jobs: list[JobRefWire] | None = None
    projects: list[str] | None = None
    active_shards: int | None = None
    needed: int | None = None
    soft_limit: int | None = None
    hard_limit: int | None = None
    heap_max_bytes: int | None = None
    current_revision: int | None = None
    # W2 (activation OCC / validation reports)
    report: ValidationReport | None = None
    current: ActiveRef | None = None
    axis: str | None = None
    valid_ids: list[str] | None = None
    # Delta 12: the full capacity object on shard_budget_exceeded (and in
    # the shard_budget_high warning), so a create form re-renders from
    # one response instead of a second GET /projects.
    capacity: ProjectCapacityWire | None = None
    # invalid_transition: the status the project is in, and the action refused.
    project_status: str | None = None
    action: str | None = None
    # W2b: 409 class_hotkey_conflict / the unbind-and-save response, and
    # 422 hotkey_reserved / 409 hotkey_taken on PUT /classes/{id}.
    class_conflicts: list[dict[str, Any]] | None = None
    unbound_class_hotkeys: list[dict[str, Any]] | None = None
    actions: list[dict[str, Any]] | None = None
    class_id: int | None = None
    class_name: str | None = None
    # W10: 422 import_blocked / dataset_path_not_allowed carry the issues,
    # 422 class_mapping_incomplete the unmapped dataset classes, a 409 or a
    # 413 the import id / byte limit it concerns.
    issues: list[DatasetIssueWire] | None = None
    unmapped: list[str] | None = None
    import_id: str | None = None
    limit: int | None = None
    # W9: unknown_vlm carries `requested`; a refused external endpoint names
    # itself and where its acknowledgement is given.
    requested: str | None = None
    endpoint: str | None = None
    activate_via: str | None = None
    # W5: 404 crop_not_found names every crop id the project does not have.
    crop_ids: list[str] | None = None
    # 409 in_use on PUT /models/{name}/sharing: who still runs the model.
    used_by: list[ModelSharingUser] | None = None


class ApiErrorResponse(BaseModel):
    """The body of every error :func:`api_error` raises."""

    detail: ConfigErrorDetail


def api_error(status: int, code: ErrorCode, message: str, **fields: Any) -> HTTPException:
    """Build ``HTTPException(status, {"detail": ConfigErrorDetail})``."""
    detail = ConfigErrorDetail(error=code, message=message, **fields)
    return HTTPException(status_code=status, detail=detail.model_dump(exclude_none=False))


class ValidationIssue(BaseModel):
    """One error/warning/info from a config validator (§3.3/§4.3)."""

    code: ValidationCode
    severity: Literal['error', 'warning', 'info']
    field: str | None = None
    message: str
    detail: dict[str, Any] = {}
    bypassable: bool = False


class ValidationReport(BaseModel):
    """The result of running a config validator (``/validate``, create,
    clone, PUT, activate). ``force_allowed`` is true only when every
    error present is individually bypassable -- the GUI's "activate
    anyway" affordance."""

    ok: bool
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    force_allowed: bool = False


class ActiveRef(BaseModel):
    """``{name, revision}`` for the currently-active config on one axis.
    ``name=None`` means the axis is off/unconfigured."""

    name: str | None = None
    revision: int | None = None


class AppliedRuntime(BaseModel):
    """One worker process's "what did I actually apply" record (any_domain_plan.md
    §4.5/§7.3), served under ``ActiveConfigResponse.applied[]`` from
    ``runtime:<process>:<host>`` docs (``upsert_project_runtime_doc``,
    W2). ``lagging`` is ``true`` when ``applied_config_revision`` is
    behind the axis's current ``config_revision`` for longer than the
    drain-plus-poll grace period (§4.5) -- a stuck/slow worker, not a
    momentary swap in progress."""

    process: str
    host: str
    applied_config_revision: int
    profile: ActiveRef
    pack: ActiveRef
    # W9: the VLM endpoint this worker's runtime was built from. ``null``
    # = the worker never reported a VLM axis (older worker); an ActiveRef
    # with ``name=None`` = it reported "no VLM configured".
    vlm: ActiveRef | None
    applied_at: str | None = None
    lagging: bool = False

    @classmethod
    def from_runtime_doc(cls, doc: dict[str, Any], *, config_revision: int) -> AppliedRuntime:
        """The record for one ``runtime:<process>:<host>`` doc, which the
        worker writes flat (``profile`` / ``profile_revision`` / ``pack`` /
        ``pack_revision`` / ``vlm`` / ``vlm_revision``). The one reader, for
        every axis's ``/active`` route."""
        applied_rev = int(doc.get('applied_config_revision') or 0)
        return cls(
            process=doc.get('process', 'detection_worker'),
            host=doc.get('host', ''),
            applied_config_revision=applied_rev,
            profile=ActiveRef(name=doc.get('profile'), revision=doc.get('profile_revision')),
            pack=ActiveRef(name=doc.get('pack'), revision=doc.get('pack_revision')),
            vlm=(
                ActiveRef(name=doc['vlm'], revision=doc.get('vlm_revision'))
                if 'vlm' in doc
                else None
            ),
            applied_at=doc.get('applied_at'),
            lagging=applied_rev < config_revision,
        )


class ActiveConfigResponse(BaseModel):
    """``GET /prompt_packs/active`` / ``GET /region_profiles/active`` (W3/W4);
    also the activation-mutation response shape used by W2's settings
    bridge and by ``store.activate``'s callers.

    ``source`` (Cropwright W3 UI, C2/Q5): where ``active`` came from --
    ``'stored'`` (an activation doc names a saved pack/profile),
    ``'env'`` (never activated through the store; the env/file default
    applies, and ``active.name`` names it), or ``'off'`` (explicitly
    deactivated -- ``active.name`` is ``None``). Never guessed from
    ``active`` alone: ``'env'`` is also nameless when no env default is
    configured, but only an explicit deactivation is ``'off'``.
    ``activated_at`` is the activation doc's own timestamp -- ``None``
    for ``'env'`` (there was no activation write). ``applied`` is every
    live ``runtime:*`` doc for this axis (§4.5) -- empty when no worker
    has ever applied anything, e.g. an API-only deployment.
    """

    axis: Literal['prompt_pack', 'detection_profile', 'vlm', 'open_vocab']
    active: ActiveRef
    source: Literal['stored', 'env', 'off']
    activated_at: str | None = None
    previous: ActiveRef | None = None
    config_revision: int
    stale: bool = False
    applied: list[AppliedRuntime] = []


class ActivateResponse(ActiveConfigResponse):
    """``POST /{prompt_packs,region_profiles}/{name}/activate``: the new
    active state plus the activation gate's validation report."""

    validation: ValidationReport


ConfigErrorDetail.model_rebuild()
