/**
 * Wire types for combine-projects (OpenProcessor P4,
 * `POST {globalApi()}/projects/combine*`; projects_plan.md §6;
 * docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md §4).
 *
 * Pinned key-for-key to the vendored OpenAPI by
 * `src/lib/contract/combineContract.test.ts`. The contract types the
 * preview's `sources` / `target` / `dedup` and the job's `report` /
 * `next_steps` as bare objects; the `CombinePreview*` / `CombineDedup` /
 * `CombineNextStep` shapes below are what the backend builds today
 * (`plan.py::build_preview`, `execute.py`), every field optional or
 * nullable so a missing one renders as "—", never a guess (plan question
 * P4-6).
 */
import type { ClassMappingEntry } from '$lib/types_import';

export type CombineLabelStates = 'all' | 'validated_only';
export type CombineDedupMode = 'content_hash' | 'none';
export type CombineHoldoutMode = 'preserve_union' | 'recompute' | 'none';

export interface CombineTarget {
  slug: string;
  display_name: string;
  description?: string;
}

export interface CombineInclude {
  label_states?: CombineLabelStates;
}

export interface CombineSource {
  project: string;
  include?: CombineInclude;
}

/** A mapping row as combine sends it: `create` defines a target class,
 *  `map` names one by `new_class_name`; no `class_id` (a class is its
 *  name). */
export type CombineMappingEntry = ClassMappingEntry;

export interface CombineRequest {
  target: CombineTarget;
  sources: CombineSource[];
  class_mapping?: Record<string, CombineMappingEntry[]>;
  target_classes?: string[] | null;
  dedup?: CombineDedupMode;
  dedup_iou?: number;
  holdout?: CombineHoldoutMode;
  settings_from?: string | null;
}

export interface CombineStartRequest extends CombineRequest {
  expected_preview_sha: string;
}

export interface CombineStartResponse {
  job_id: string;
  target: string;
}

export interface CombineIssue {
  code: string;
  severity?: 'error' | 'warning';
  project?: string | null;
  message?: string;
  detail?: Record<string, unknown>;
}

export interface CombinePreviewClass {
  name: string;
  count: number;
  /** The served target class name, or the kind (`skip` / `region`);
   *  `null` = unmapped. */
  mapped_to: string | null;
}

export interface CombinePreviewSource {
  project: string;
  images?: number | null;
  items?: number | null;
  labeled_items?: number | null;
  holdout_images?: number | null;
  classes?: CombinePreviewClass[];
}

export interface CombinePreviewTargetClass {
  id: number;
  name: string;
  count: number;
  from?: { project: string; class: string }[];
}

export interface CombinePreviewTarget {
  slug?: string;
  slug_available?: boolean;
  classes?: CombinePreviewTargetClass[];
  images?: number | null;
  items?: number | null;
  holdout_images?: number | null;
}

export interface CombineDedup {
  identical_images?: number | null;
  merged_items?: number | null;
  conflicts?: number | null;
  conflict_samples?: Record<string, unknown>[];
  near_duplicate_pairs_estimate?: number | null;
}

export interface CombinePreview {
  ok: boolean;
  errors: CombineIssue[];
  warnings: CombineIssue[];
  preview_sha: string;
  suggested_mapping: Record<string, CombineMappingEntry[]>;
  sources: CombinePreviewSource[];
  target: CombinePreviewTarget;
  dedup: CombineDedup;
  bytes: { to_link?: number; to_copy?: number };
}

/** One served follow-up of a finished job: its `method` against its
 *  `path` under the target project's prefix, body-less. */
export interface CombineNextStep {
  action: string;
  method: string;
  path: string;
  reason?: string;
}

export interface CombineJobResponse {
  job_id: string;
  status: string;
  phase?: string | null;
  done?: number;
  total?: number;
  started_at?: string | null;
  finished_at?: string | null;
  sources?: string[];
  target?: string | null;
  error?: string | null;
  report?: Record<string, unknown>;
  next_steps?: CombineNextStep[];
}

/** The slice of `GET {project prefix}/datasets/formats` combine reads:
 *  the served mapping-action labels (contract `LabeledChoice`). */
export interface CombineMappingActionChoice {
  value: string;
  label: string;
  description?: string;
}

export interface CombineFormatsVocabulary {
  mapping_actions: CombineMappingActionChoice[];
}
