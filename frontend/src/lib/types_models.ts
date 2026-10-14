/**
 * Wire types for cross-project model sharing (OpenProcessor projects P2,
 * `projects_plan.md` §5.5). A promoted model belongs to one project; its
 * owner may opt it into sharing, and its classes reach another project
 * by NAME only — class ids never cross a project boundary, so nothing
 * here renders one.
 *
 * Pinned key-for-key to the vendored OpenAPI by
 * `src/lib/contract/modelsContract.test.ts`, except
 * `ModelClassMappingSummary`: `GET {scoped}/models/status` is untyped in
 * the served OpenAPI, so that per-entry shape is documented here, from
 * the backend's `ModelClassMappingSummary`.
 */

/** `class_mapping` on a `GET {scoped}/models/status` entry: how many of
 *  the model's classes map by name onto the active project's registry,
 *  and the model class names that don't. `null` on the entry for a model
 *  with no class list (an encoder, an OCR model, an external service). */
export interface ModelClassMappingSummary {
  mapped_count: number;
  unmapped: string[];
}

/** `PUT {scoped}/models/{name}/sharing` body. `expected_revision` is the
 *  model's served sharing revision (optimistic concurrency). */
export interface ModelSharingRequest {
  shared: boolean;
  expected_revision: number;
}

export interface ModelSharingUser {
  project: string;
  profile?: string | null;
}

/** `PUT {scoped}/models/{name}/sharing` 409 `in_use` detail
 *  (`ModelInUseDetail`): another project's active detection profile uses
 *  the model. `used_by` carries each project with its served profile. */
export interface ModelInUseDetail {
  error: 'in_use';
  message: string;
  projects: string[];
  used_by: ModelSharingUser[];
}

/** 409 `revision_conflict` detail (`ModelRevisionConflictDetail`): a stale
 *  `expected_revision`. */
export interface ModelRevisionConflictDetail {
  error: 'revision_conflict';
  message: string;
  current_revision: number;
}

/** `{detail}` of the 409 (`ModelSharingConflictResponse`). */
export type ModelSharingConflictDetail = ModelInUseDetail | ModelRevisionConflictDetail;

/** 503 detail (`ModelSharingUnavailableDetail`): another project's config
 *  could not be read. */
export interface ModelSharingUnavailableDetail {
  error: 'config_store_unavailable';
  message: string;
}

/** `PUT {scoped}/models/{name}/sharing` 200. `used_by` lists the other
 *  projects whose ACTIVE detection profile uses the model (served since
 *  OpenProcessor f14f4ddc; non-empty here only on a forced unshare). */
export interface ModelSharingResponse {
  name: string;
  project: string;
  shared: boolean;
  revision: number;
  used_by?: ModelSharingUser[];
}

export type ModelClassMatch = 'exact' | 'case_insensitive' | 'none';

export interface ModelClassMappingEntry {
  model_id: number;
  model_name: string;
  class_id: number | null;
  class_name: string | null;
  match: ModelClassMatch;
}

export interface ModelClassMappingLabels {
  /** Display copy for every served `match` kind. */
  match?: Record<string, string>;
}

/** `GET {scoped}/models/{name}/class_mapping` — the model's classes
 *  matched by name onto the active project's registry. */
export interface ModelClassMappingResponse {
  model: string;
  model_project: string | null;
  project: string;
  entries: ModelClassMappingEntry[];
  unmapped: string[];
  /** Project classes the model can't predict (informational). */
  not_covered: string[];
  labels?: ModelClassMappingLabels;
}

export type ModelStatus =
  'ready' | 'not_ready' | 'unavailable' | 'not_configured' | 'not_installed';
export type ModelKind = 'triton' | 'vlm' | 'external';

export interface ModelInfo {
  name: string;
  friendly_name: string;
  role: string;
  kind: ModelKind;
  model_type: string;
  status: ModelStatus;
  version: string | null;
  inference_count: number | null;
  exec_count: number | null;
  inference_failed: number | null;
  avg_latency_ms: number | null;
  last_error: string | null;
  endpoint: string | null;
  /**
   * Follow-up gap 2 (docs/design/audit-remediation-plan-2026-09.md
   * Appendix D item 3, 2026-09-11): `DELETE {API_PREFIX}/models/{name}` guard
   * flags, mirrored from the same checks the endpoint enforces
   * server-side so the UI never has to re-derive them (and can't drift
   * out of sync with the real guard).
   */
  /** Region-detection pipeline model (backend `_is_region_protected_model`) — never unloadable through the UI, no override. */
  is_region_protected?: boolean;
  /** The active production model or another core pipeline model — unload requires force=true. */
  requires_force_to_unload?: boolean;
  /** Served directly (2026-09-25 follow-up to #36 item 5) — `false` for
   *  every external-service entry (the segmenter, the VLM) and any model
   *  the active config hard-blocks (`is_region_protected`, now also
   *  covering the ingest primary proposer/secondary classifier and the
   *  OCR det/rec pair, not just the region detector). `unloadButtonState`
   *  reads this FIRST, ahead of `is_region_protected` — the server's own
   *  verdict, never re-derived from the other flags. */
  unloadable: boolean;
  /** Served (OpenProcessor ba88751): true only for a model the pipeline
   *  can run without — the region profile's detector when a segmenter is
   *  configured. Pairs with status `not_installed`. */
  optional: boolean;
  /** Present (with job_id/version) only for models promoted through this pipeline. */
  job_id?: string | null;
  promoted_at?: string | null;
  /** Served (projects P2, §5.5): the owning project's slug (a promoted
   *  model's `promote.json.project`); `null` for a model with no
   *  promote.json and for every external service. */
  project: string | null;
  /** Served: whether the owner opted this model into cross-project
   *  sharing. Another project's model is only ever listed when true. */
  shared: boolean;
  /** Served: the model's classes matched by name onto the active
   *  project's registry; `null` for a model with no class list. */
  class_mapping: ModelClassMappingSummary | null;
  /** Served: whether THIS project owns the model (the route's own
   *  ownership check) — the only thing the owner-only sharing toggle
   *  reads. `false` for another project's shared model, a base model with
   *  no promote.json and every external service. */
  owned: boolean;
  /** Served: the model's sharing revision, sent back as
   *  `expected_revision` on `PUT .../sharing`. Non-null only when `owned`. */
  sharing_revision: number | null;
  /** Served on a VLM row (`kind: 'vlm'`, one per registered endpoint):
   *  whether this endpoint is the project's active one, and which
   *  projects the listing names as running it (the bound project only on
   *  this scoped route). */
  active?: boolean;
  active_in?: string[];
  /** Served on a VLM row: the endpoint's model id as the server resolves
   *  it (the row's `name` is the endpoint name). */
  model?: string | null;
}

export interface ModelsStatus {
  models: ModelInfo[];
}

export interface UnloadModelResponse {
  triton_name: string;
  triton_unloaded: boolean;
  directory_removed: boolean;
  forced: boolean;
  warning: string | null;
}
