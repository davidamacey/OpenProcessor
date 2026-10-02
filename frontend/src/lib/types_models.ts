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
