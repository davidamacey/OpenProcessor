/**
 * Embedding spaces (`{prefix}/embedding/spaces`): the vector spaces a project
 * holds per target. Search takes a ready space key (`space`) and more keys to
 * fuse (`fuse`); the response echoes both. Pinned to the OpenAPI by
 * `contract/searchSpaceContract.test.ts`.
 */

export const EMBEDDING_TARGETS = ['item', 'box', 'frame', 'item_text'] as const;
export type EmbeddingTarget = (typeof EMBEDDING_TARGETS)[number];

export const EMBEDDING_SPACE_ROLES = ['primary', 'optional', 'candidate'] as const;
export type EmbeddingSpaceRole = (typeof EMBEDDING_SPACE_ROLES)[number];

export const EMBEDDING_SPACE_STATUSES = [
  'pending',
  'building',
  'ready',
  'retiring',
  'retired',
  'failed',
] as const;
export type EmbeddingSpaceStatus = (typeof EMBEDDING_SPACE_STATUSES)[number];

export interface EmbeddingSpaceView {
  space_key: string;
  target: EmbeddingTarget;
  model_key: string;
  model_version: number;
  dim: number;
  role: EmbeddingSpaceRole;
  status: EmbeddingSpaceStatus;
  ready_at: string | null;
  docs_with_vector: number | null;
  /** The newest re-index run of the space, whoever started it; null when it never had one. */
  run_id: string | null;
  run_mode: string | null;
  /** That run's job status. */
  run_status: string | null;
}

/** `GET /embedding/spaces/{key}/evaluations`: one run of the in-product evaluation of a space. */
export interface EmbeddingEvaluationView {
  id: string;
  baseline_space_key: string;
  candidate_space_key: string;
  protocol: string;
  verdict: string;
  metrics: Record<string, unknown>;
  sample_spec: Record<string, unknown>;
  /** The baseline is the current primary: the run a promote asks for. */
  counts_for_promote: boolean;
  job_id: string | null;
  created_at: string;
  created_by: string;
}

export interface EmbeddingEvaluationList {
  space_key: string;
  target: EmbeddingTarget;
  primary_space_key: string | null;
  evaluated_against_primary: boolean;
  evaluations: EmbeddingEvaluationView[];
}

export interface EmbeddingSpaceListResponse {
  spaces: EmbeddingSpaceView[];
}

/** `GET /curation/embedding/models`: the encoders the deployment can serve (deployment-wide). */
export interface EmbeddingModelEntry {
  model_key: string;
  version: number;
  modalities: string[];
  native_dim: number;
  matryoshka_dims: number[];
  licence: string;
  available: boolean;
  unavailable_reason?: string | null;
}

export interface EmbeddingModelListResponse {
  models: EmbeddingModelEntry[];
}

/** `POST /embedding/spaces`: `dim: null` is the model's native size. */
export interface EmbeddingSpaceRegisterRequest {
  model_key: string;
  model_version: number;
  target: EmbeddingTarget;
  dim?: number | null;
}

export const EMBEDDING_BUILD_MODES = ['build', 'backfill'] as const;
export type EmbeddingBuildMode = (typeof EMBEDDING_BUILD_MODES)[number];

export interface EmbeddingBuildRequest {
  mode?: EmbeddingBuildMode;
  batch_size?: number;
}

export interface EmbeddingPromoteRequest {
  force?: boolean;
}

/** The 202 of register, build and retire: the space and the job that works on it. */
export interface EmbeddingSpaceStartResponse {
  space_key: string;
  target: EmbeddingTarget;
  job_id: string | null;
  run_id: string | null;
  bytes_estimate: number | null;
}

export interface EmbeddingPromoteResponse {
  primary: EmbeddingSpaceView;
  previous: EmbeddingSpaceView | null;
  config_epoch: number;
}

/** `GET /embedding/reindex-runs/{run_id}`: the run with its job's status and progress. */
export interface EmbeddingReindexRun {
  id: string;
  space_key: string;
  mode: string;
  scope: Record<string, unknown>;
  docs_total: number | null;
  docs_done: number;
  docs_failed: number;
  docs_skipped: number;
  bytes_estimate: number | null;
  started_at: string | null;
  finished_at: string | null;
  job_id: string;
  job_status: string | null;
  job_progress: Record<string, unknown>;
}

export interface EmbeddingRunCancelResponse {
  run_id: string;
  job_status: string;
  abandoned: boolean;
  space_status: string;
  clear_job_id?: string | null;
}
