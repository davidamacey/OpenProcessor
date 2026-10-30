/**
 * Wire shapes of the auto-label loop (`{prefix}/assist/loops`). Every number,
 * decision and refusal is the server's; the views print them.
 */

export type AssistLoopStatus =
  | 'dry_run'
  | 'queued'
  | 'running'
  | 'paused'
  | 'completed'
  | 'cancelled'
  | 'failed'
  | 'interrupted';

export type AssistLoopDecision =
  'pass' | 'split' | 'review_leaf' | 'review_small' | 'stale' | 'unvisited';

export interface AssistLoopRequest {
  scope: { cluster_ids: number[] | null; min_root_size: number };
  vlm: string | null;
  prompt_pack: string | null;
  acknowledge_external: boolean;
  seed: number;
  gate: { theta: number; alpha: number; max_dissent: number };
  recursion: { max_depth: number; branching: number; min_node_size: number };
  budget: {
    max_vlm_crops: number;
    max_vlm_requests: number | null;
    max_wall_s: number | null;
  } | null;
  audit: {
    vlm: string | null;
    acknowledge_external: boolean;
    rate: number;
    min_per_batch: number;
    max_crops: number;
    floor: number;
    on_fail: 'pause' | 'rollback';
  };
  vlm_concurrency: number;
  node_concurrency: number;
  dry_run: boolean;
}

export interface LoopVlmPin {
  name: string | null;
  revision: number | null;
  model: string | null;
}

export interface LoopRequiredSample {
  n_members: number;
  required_by_dissent: number[];
}

export interface LoopPlan {
  roots: number;
  eligible_items: number;
  required_sample_table: LoopRequiredSample[];
  estimated_vlm_crops_root_level: number;
  estimated_vlm_requests_root_level: number;
  fits_budget: boolean | null;
}

export interface LoopPoolBudget {
  max_crops: number;
  max_requests: number;
  crops_spent: number;
  crops_reserved: number;
  requests_spent: number;
  requests_reserved: number;
}

export interface LoopLevel {
  level: number;
  nodes: number;
  passed: number;
  split: number;
  review_leaf: number;
  review_small: number;
  stale: number;
  unvisited: number;
  items_propagated: number;
  items_direct: number;
  crops_spent: number;
}

export interface LoopTotals {
  items_propagated: number;
  items_direct: number;
  clusters_left_for_review: number;
  items_left_for_review: number;
  skipped: Record<string, number>;
}

export interface LoopAuditView {
  status: string;
  sampled: number;
  agreed: number;
  agreement: number | null;
  lower_bound: number | null;
  crops_spent: number;
}

export interface AssistLoopState {
  loop_id: string | null;
  status: AssistLoopStatus;
  stop_reason: string | null;
  pause_reason: string | null;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  updated_at: string | null;
  error: string | null;
  request: AssistLoopRequest;
  pinned: {
    vlm: LoopVlmPin;
    prompt_pack: { name: string | null; revision: number | null };
    audit_vlm: LoopVlmPin | null;
  };
  plan: LoopPlan;
  partition: {
    kind: string;
    digest: string;
    snapshot_at: string;
    roots: number;
    eligible_items: number;
  } | null;
  budget: { vlm: LoopPoolBudget; audit: LoopPoolBudget | null; wall_s: number } | null;
  levels: LoopLevel[];
  totals: LoopTotals | null;
  audit: LoopAuditView | null;
}

export interface LoopNode {
  path: string;
  level: number;
  parent: string | null;
  n: number;
  sampled: number;
  answered: number;
  decision: AssistLoopDecision | null;
  reason: string | null;
  class_name: string | null;
  agree: number | null;
  bound: number | null;
  distribution: { class_name: string; votes: number }[];
  new_class_proposals: { name: string; votes: number }[];
  items_propagated: number;
}

export interface LoopNodePage {
  nodes: LoopNode[];
  next_after: string | null;
}

export interface LoopMembersPage {
  crop_ids: string[];
  next_after: string | null;
}

export interface LoopRollbackResponse {
  dry_run: boolean;
  matched: number;
  restored: number;
  kept_changed: number;
  conflicts: number;
}

/** The loop holds the project's cluster training and the refresh daemon while in these. */
export const LIVE_LOOP_STATUSES: AssistLoopStatus[] = ['queued', 'running'];
