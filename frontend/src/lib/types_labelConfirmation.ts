// Label confirmation (#119): wire types for the `VlmPolicy*` and `Audit*`
// schemas of the contract (`contracts/openapi/curation.json`), pinned key
// for key by `contract/labelConfirmationContract.test.ts`.

export const VLM_SCOPES = ['all', 'uncertain', 'representatives', 'off'] as const;
export type VlmScope = (typeof VLM_SCOPES)[number];

/** `VlmPolicy`: which crops the VLM labels, and its daily budget. */
export interface VlmPolicy {
  scope?: VlmScope;
  conf_max?: number;
  per_cluster?: number;
  max_crops_per_day?: number;
  sample_frac?: number;
  revision?: number;
}

/** The editable part of a policy (everything but `revision`). */
export type VlmPolicyBody = Omit<VlmPolicy, 'revision'>;

/** `VlmPolicyUpdate`: the `PUT` body. */
export interface VlmPolicyUpdate extends VlmPolicyBody {
  expected_revision: number;
}

/** `AuditStartRequest`. */
export interface AuditStartRequest {
  min_per_class?: number;
  sample_size?: number;
}

/** `AuditStratum`. */
export interface AuditStratum {
  detector_class: string;
  available: number;
  sampled: number;
  short_of_floor: boolean;
}

/** `AuditStartResponse`. */
export interface AuditStartResponse {
  batch_id: string;
  min_per_class: number;
  requested: number;
  sampled: number;
  strata: AuditStratum[];
}

/** `AuditClassStat`: precision of one class with its Wilson 95% interval. */
export interface AuditClassStat {
  name: string;
  n: number;
  correct: number;
  precision: number | null;
  ci_low: number;
  ci_high: number;
  insufficient_sample: boolean;
}

/** `AuditReport`. */
export interface AuditReport {
  audited: number;
  pending: number;
  min_per_class: number;
  detector: AuditClassStat[];
  vlm: AuditClassStat[];
  /** `{detector class: {human class: crops}}`. */
  confusion: Record<string, Record<string, number>>;
  /** agree / detector_wrong / vlm_wrong / both_wrong. */
  outcomes: Record<string, number>;
}
