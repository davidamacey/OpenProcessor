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

export const NEAR_DUP_PURPOSES = ['dedup', 'leak'] as const;
export type NearDupPurpose = (typeof NEAR_DUP_PURPOSES)[number];

export const NEAR_DUP_DECISIONS = ['same', 'different'] as const;
export type NearDupDecision = (typeof NEAR_DUP_DECISIONS)[number];

export interface NearDupRunPage {
  items: NearDupRunView[];
}

export interface NearDupGroupSummary {
  /** The lineage id: stable while the group keeps its members. */
  group_id: string;
  purpose: NearDupPurpose;
  size: number;
  representative_id: string;
  min_edge_cosine: number;
  min_cosine_to_rep: number;
  has_holdout: boolean;
  /** The representative's served picture. */
  representative_thumbnail_url: string;
}

export interface NearDupGroupPage {
  items: NearDupGroupSummary[];
  next_cursor: string | null;
  /** Why a first page is empty (the cut and the number scored); null otherwise. */
  empty_reason: string | null;
}

/** The pin between a member and the representative; `id` undoes it. */
export interface NearDupPin {
  id: string;
  kind: NearDupDecision;
}

export interface NearDupMemberView {
  /** An image id at the image level, an item id at the item level. */
  member_id: string;
  is_representative: boolean;
  cosine_to_rep: number;
  thumbnail_url: string;
  image_id: string | null;
  crop_id: string | null;
  class_name: string | null;
  in_holdout: boolean;
  decision: NearDupPin | null;
}

export interface NearDupGroupDetail {
  group: NearDupGroupSummary;
  members: NearDupMemberView[];
}

export interface NearDupDecisionIn {
  level: NearDupLevel;
  a_id: string;
  b_id: string;
  decision: NearDupDecision;
}

export interface NearDupDecisionView extends NearDupDecisionIn {
  id: string;
  decided_by: string;
  decided_at: string;
  a_thumbnail_url: string;
  b_thumbnail_url: string;
}

export interface NearDupDecisionPage {
  items: NearDupDecisionView[];
  next_cursor: string | null;
}

export interface NearDupDecisionQuery {
  level?: NearDupLevel;
  member_id?: string;
  cursor?: string;
  limit?: number;
}

export interface NearDupGroupQuery {
  level: NearDupLevel;
  purpose?: NearDupPurpose;
  min_size?: number;
  has_holdout?: boolean;
  cursor?: string;
  limit?: number;
}

/** `ResearchActive` of `GET /near_dup_profiles/active`. */
export interface NearDupActiveProfile {
  axis: string;
  state: 'none' | 'active' | 'off';
  name?: string | null;
  revision?: number | null;
  activated_at?: string | null;
}

/** The cuts both levels share (`NearDupProfileBody.image` / `.item`). */
export interface NearDupLevelCuts {
  space_key?: string | null;
  dedup_cosine: number;
  leak_cosine: number;
  max_group_size: number;
  [key: string]: unknown;
}

export interface NearDupImageCuts extends NearDupLevelCuts {
  k: number;
}

export interface NearDupItemCuts extends NearDupLevelCuts {
  scope: string;
  same_detector_class: boolean;
}

export interface NearDupProfileBody {
  version: number;
  image: NearDupImageCuts;
  item: NearDupItemCuts;
}

/** `ResearchConfigDoc` of a `near_dup_profile`. */
export interface NearDupProfileDoc {
  kind: string;
  name: string;
  revision: number;
  description: string;
  body: NearDupProfileBody;
  created_at: string;
  updated_at: string;
  updated_by: string;
}
