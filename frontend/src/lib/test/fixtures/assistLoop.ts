/** Served-shaped auto-label loop states for the controller and component tests. */
import type { AssistLoopDeps } from '$lib/assistLoop/assistLoopController.svelte';
import type { AssistLoopRequest, AssistLoopState, LoopNode } from '$lib/types_assistLoop';

export const LOOP_REQUEST: AssistLoopRequest = {
  scope: { cluster_ids: null, min_root_size: 40 },
  vlm: null,
  prompt_pack: null,
  acknowledge_external: false,
  seed: 0,
  gate: { theta: 0.9, alpha: 0.05, max_dissent: 1 },
  recursion: { max_depth: 3, branching: 4, min_node_size: 40 },
  budget: { max_vlm_crops: 2000, max_vlm_requests: null, max_wall_s: null },
  audit: {
    vlm: null,
    acknowledge_external: false,
    rate: 0.01,
    min_per_batch: 30,
    max_crops: 500,
    floor: 0.85,
    on_fail: 'pause',
  },
  vlm_concurrency: 16,
  node_concurrency: 8,
  dry_run: false,
};

export function loopState(
  status: AssistLoopState['status'],
  over: Partial<AssistLoopState> = {},
): AssistLoopState {
  const started = status !== 'dry_run';
  return {
    loop_id: started ? 'loop_abc123' : null,
    status,
    stop_reason: null,
    pause_reason: null,
    created_at: '2026-10-29T10:00:00+00:00',
    started_at: started ? '2026-10-29T10:00:05+00:00' : null,
    finished_at: null,
    updated_at: null,
    error: null,
    request: { ...LOOP_REQUEST, dry_run: status === 'dry_run' },
    pinned: {
      vlm: { name: 'local_vlm', revision: 3, model: 'qwen3-vl-4b' },
      prompt_pack: { name: 'default', revision: 1 },
      audit_vlm: null,
    },
    plan: {
      roots: 12,
      eligible_items: 3400,
      required_sample_table: [
        { n_members: 300, required_by_dissent: [32, 49] },
        { n_members: 80, required_by_dissent: [25, 40] },
      ],
      estimated_vlm_crops_root_level: 380,
      estimated_vlm_requests_root_level: 95,
      fits_budget: true,
    },
    partition: null,
    budget: started
      ? {
          vlm: {
            max_crops: 2000,
            max_requests: 600,
            crops_spent: 500,
            crops_reserved: 8,
            requests_spent: 130,
            requests_reserved: 2,
          },
          audit: null,
          wall_s: 75,
        }
      : null,
    levels: started
      ? [
          {
            level: 0,
            nodes: 12,
            passed: 7,
            split: 5,
            review_leaf: 0,
            review_small: 0,
            stale: 0,
            unvisited: 0,
            items_propagated: 2100,
            items_direct: 224,
            crops_spent: 400,
          },
        ]
      : [],
    totals: started
      ? {
          items_propagated: 2100,
          items_direct: 224,
          clusters_left_for_review: 1,
          items_left_for_review: 90,
          skipped: { dissent: 0, drifted: 3, locked: 12 },
        }
      : null,
    audit: started
      ? {
          status: 'off',
          sampled: 0,
          agreed: 0,
          agreement: null,
          lower_bound: null,
          crops_spent: 0,
        }
      : null,
    ...over,
  };
}

export function loopNode(over: Partial<LoopNode> = {}): LoopNode {
  return {
    path: 'r10001',
    level: 0,
    parent: null,
    n: 300,
    sampled: 33,
    answered: 33,
    decision: 'pass',
    reason: null,
    class_name: 'widget_a',
    agree: 33,
    bound: 0.913,
    distribution: [{ class_name: 'widget_a', votes: 33 }],
    new_class_proposals: [],
    items_propagated: 267,
    ...over,
  };
}

/** Deps that answer from the given state; every call is a vi.fn-compatible plain function. */
export function loopDeps(over: Partial<AssistLoopDeps> = {}): AssistLoopDeps {
  const state = loopState('completed');
  return {
    create: async (body) => loopState(body.dry_run ? 'dry_run' : 'queued'),
    list: async () => ({ loops: [state] }),
    get: async () => state,
    nodes: async () => ({ nodes: [loopNode()], next_after: null }),
    members: async () => ({ crop_ids: ['c1', 'c2'], next_after: null }),
    cancel: async () => loopState('cancelled'),
    resume: async () => loopState('queued'),
    rollback: async (_id, body) => ({
      dry_run: body.dry_run,
      matched: 267,
      restored: 260,
      kept_changed: 7,
      conflicts: 0,
    }),
    debounceMs: 0,
    pollMs: 5,
    ...over,
  };
}
