/**
 * Wire types for OpenProcessor v0.4.0's SAM 3 open-vocabulary sets
 * (`{prefix}/open_vocab*`) and region stage (`{prefix}/region_stage*`).
 * Pinned key-for-key to the vendored OpenAPI (fce17771) by
 * `contract/openVocabContract.test.ts` and `contract/regionStageContract.test.ts`.
 *
 * `OpenVocabBody` and its parts are `additionalProperties: true` on the
 * wire; the keys listed are the declared ones, and every one is optional
 * on the way in (the server fills defaults).
 */
import type {
  ActiveConfigResponse,
  ActiveRef,
  AppliedRuntime,
  ConfigDocBase,
  ValidationReport,
} from './types_config';
import type { OpenVocabStatus, ReprocessRequest } from './types_import';

export type { OpenVocabStatus };

export type OpenVocabTargetBody = {
  prompt?: string;
  class_name?: string;
  enabled?: boolean;
  mask?: boolean;
  min_score?: number;
  min_area_frac?: number;
  max_area_frac?: number;
  max_instances?: number;
  parent_classes?: string[];
};

export type OpenVocabHitRateBody = {
  enabled?: boolean;
  miss_threshold?: number;
  sample_floor?: number;
  window?: number;
};

export type OpenVocabGatingBody = {
  tier2_vlm_precheck?: boolean;
  tier3_hit_rate?: OpenVocabHitRateBody;
};

export type OpenVocabBody = {
  display_name?: string;
  run_on_ingest?: boolean;
  image_max_side?: number;
  dedup_iou?: number;
  max_enabled_targets?: number;
  targets?: OpenVocabTargetBody[];
  gating?: OpenVocabGatingBody;
};

/** `OpenVocabDoc`. The shared base also types `updated_by`, which this
 *  resource's schema does not serve (it reads `undefined`; nothing shows it). */
export interface OpenVocabDoc extends ConfigDocBase<OpenVocabBody> {
  source: 'stored' | 'template';
}

export interface OpenVocabSummary {
  name: string;
  source: 'stored' | 'template';
  read_only: boolean;
  revision: number | null;
  etag: string;
  display_name: string;
  n_targets: number;
  n_enabled_targets: number;
  run_on_ingest: boolean;
  active: boolean;
  active_revision?: number | null;
  updated_at?: string | null;
}

export interface OpenVocabTemplateSummary {
  name: string;
  path: string;
  n_targets: number;
  display_name?: string | null;
  read_only?: true;
  source?: 'template';
}

/** `SegmenterAvailability`: the segmenter every pass needs. `reachable` is
 *  never true when it is not configured. */
export interface SegmenterAvailability {
  configured: boolean;
  reachable: boolean;
}

export interface OpenVocabList {
  sets: OpenVocabSummary[];
  templates: OpenVocabTemplateSummary[];
  active: ActiveRef;
  config_revision: number;
  stale?: boolean;
  segmenter: SegmenterAvailability;
}

export type OpenVocabFieldScope = 'set' | 'target' | 'gating' | 'tier3_hit_rate';
export type OpenVocabFieldType = 'string' | 'int' | 'float' | 'bool' | 'string_list';

export interface OpenVocabFieldSchema {
  scope: OpenVocabFieldScope;
  field: string;
  label: string;
  type: OpenVocabFieldType;
  default: unknown;
  advanced?: boolean;
  help?: string;
  min?: number | null;
  max?: number | null;
}

/** `VocabularyOption`: one served value and its label. */
export interface VocabularyOption {
  value: string;
  label: string;
}

/** `OpenVocabVocabulary`: served labels for the pass's closed value sets
 *  (an image's `open_vocab_status`, a hit's `drop_reason`, a gate skip's
 *  `reason`). */
export interface OpenVocabVocabulary {
  statuses: VocabularyOption[];
  drop_reasons: VocabularyOption[];
  gate_reasons: VocabularyOption[];
}

export interface OpenVocabSchema {
  fields: OpenVocabFieldSchema[];
  max_enabled_targets_ceiling: number;
  vocabulary: OpenVocabVocabulary;
}

export interface OpenVocabRevisionSummary {
  revision: number;
  saved_at: string | null;
  cloned_from: string | null;
  description: string;
}

export interface OpenVocabRevisionsResponse {
  name: string;
  revisions: OpenVocabRevisionSummary[];
}

export interface OpenVocabCreateRequest {
  name: string;
  body: OpenVocabBody;
  description?: string;
}

export interface OpenVocabSaveRequest {
  expected_revision: number;
  body: OpenVocabBody;
  description?: string | null;
}

export interface OpenVocabCloneRequest {
  new_name: string;
  revision?: number | null;
  source?: 'stored' | 'template' | null;
  description?: string | null;
  from_project?: string | null;
}

export interface OpenVocabActivateRequest {
  revision?: number | null;
  expected_active?: ActiveRef | null;
  force?: boolean;
}

export interface OpenVocabRollbackRequest {
  expected_active?: ActiveRef | null;
}

export interface OpenVocabDeactivateRequest {
  expected_active?: ActiveRef | null;
}

export interface OpenVocabValidateRequest {
  name?: string | null;
  body: OpenVocabBody;
}

export interface OpenVocabActivateResponse extends ActiveConfigResponse {
  validation: ValidationReport;
  applied: AppliedRuntime[];
}

/** One unsaved target on exactly one of `image_id` / `image_base64`. */
export interface OpenVocabTestRequest {
  target: OpenVocabTargetBody;
  image_id?: string | null;
  image_base64?: string | null;
  image_max_side?: number;
  dedup_iou?: number;
  gating?: OpenVocabGatingBody;
}

export type OpenVocabDropReason =
  | 'below_min_score'
  | 'too_small'
  | 'too_large'
  | 'nms'
  | 'over_max'
  | 'cross_target_nms'
  | 'agree_existing'
  | 'skipped_locked';

export type OpenVocabGateReason = 'disabled' | 'no_parent_class' | 'vlm_no' | 'hit_rate';

export interface OpenVocabTestHit {
  bbox_norm: number[];
  score: number;
  selected: boolean;
  drop_reason?: OpenVocabDropReason | null;
  mask_polygon?: number[][] | null;
}

export interface OpenVocabTestGate {
  run: boolean;
  tier?: number | null;
  reason?: OpenVocabGateReason | null;
}

export interface OpenVocabTestImage {
  width: number;
  height: number;
}

export interface OpenVocabTestResponse {
  image: OpenVocabTestImage;
  prompt: string;
  class_name: string;
  gate: OpenVocabTestGate;
  hits: OpenVocabTestHit[];
  elapsed_ms: number;
  validation: ValidationReport;
}

export interface RegionStageCounts {
  pending_detection: number;
  pending_verification: number;
  gate_skipped: number;
}

export interface RegionStageState {
  project: string;
  paused: boolean;
  paused_since?: string | null;
  pipeline_paused: boolean;
  counts: RegionStageCounts;
  /** The served request that re-runs the gate-skipped items; sent as served. */
  rerun_skipped: ReprocessRequest;
}
