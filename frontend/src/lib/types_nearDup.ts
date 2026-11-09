/**
 * Near-duplicate wire types (`{prefix}/near_duplicates/*`) and the 409
 * `near_dup_stale` refusal of the holdout freeze and the training exports.
 * Pinned to the OpenAPI by `contract/nearDupContract.test.ts`.
 */

export const NEAR_DUP_LEVELS = ['image', 'item'] as const;
export type NearDupLevel = (typeof NEAR_DUP_LEVELS)[number];

export const NEAR_DUP_STALE_REASONS = ['pending', 'rebuild_required', 'no_run'] as const;
export type NearDupStaleReason = (typeof NEAR_DUP_STALE_REASONS)[number];

export const NEAR_DUP_RUN_MODES = ['full', 'incremental'] as const;
export type NearDupRunMode = (typeof NEAR_DUP_RUN_MODES)[number];

export interface NearDupRunView {
  id: string;
  level: string;
  mode: string;
  status: string;
  space: string;
  algorithm_version: string;
  profile_revision: number | null;
  params: Record<string, unknown>;
  members_scored: number;
  grouped_members: number;
  dedup_groups: number;
  leak_groups: number;
  largest_group: number;
  oversized_split: number;
  holdout_absorbed: number;
  started_at: string;
  finished_at: string | null;
}

export interface NearDupLevelStatus {
  level: NearDupLevel;
  current_run: NearDupRunView | null;
  pending_count: number;
  stale_reasons: NearDupStaleReason[];
}

export interface NearDupStatus {
  profile: { name: string; revision: number } | null;
  levels: NearDupLevelStatus[];
}

export interface NearDupRunRequest {
  level: NearDupLevel;
  mode: NearDupRunMode;
}

export interface NearDupRunStarted {
  job_id: string;
  kind: string;
}

/** One level named by a 409 `near_dup_stale` body. */
export interface NearDupStaleLevel {
  level: NearDupLevel;
  stale_reasons: NearDupStaleReason[];
  pending_count: number;
}
