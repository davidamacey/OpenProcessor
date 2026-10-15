/** Served-shaped assist run states for the controller and component tests. */
import type {
  AssistProposal,
  AssistRunRequest,
  AssistRunState,
  AssistStatus,
} from '$lib/types_assist';

export const CLUSTER_REQUEST: AssistRunRequest = {
  target: { kind: 'cluster', cluster_id: 10000, cluster_subid: '10000b' },
  sample: { mode: 'representative', size: null, seed: 0, stratify_by: 'auto' },
  actions: ['classify'],
  dry_run: true,
};

export function proposal(over: Partial<AssistProposal> = {}): AssistProposal {
  return {
    class_name: 'widget_a',
    in_registry: true,
    agreement: 0.83,
    answered: 12,
    min_agreement: 0.6,
    passes_gate: true,
    distribution: [
      { class_name: 'widget_a', votes: 10 },
      { class_name: 'widget_b', votes: 2 },
    ],
    new_class_proposals: [{ name: 'widget_c', votes: 1 }],
    attribute_summary: [
      {
        key: 'finish',
        answered: 11,
        values: [
          { value: 'matte', count: 8 },
          { value: 'gloss', count: 3 },
        ],
      },
    ],
    ...over,
  };
}

export function assistRun(
  status: AssistStatus = 'dry_run',
  over: Partial<AssistRunState> = {},
): AssistRunState {
  return {
    run_id: status === 'dry_run' ? null : 'asr_1',
    status,
    request: { ...CLUSTER_REQUEST, dry_run: status === 'dry_run' },
    resolved: {
      scope_count: 412,
      scope_digest: 'sha256:abc',
      sampled: 41,
      sample_seed: 0,
      strata: [
        { key: '10000a', size: 120, sampled: 12 },
        { key: '10000b', size: 292, sampled: 29 },
      ],
      skipped: { holdout: 3, excluded: 1, class_locked: 27 },
      vlm: { name: 'env', revision: 3, model: 'vlm-model' },
      prompt_pack: { name: 'generic', revision: 7 },
      estimated_vlm_calls: 6,
    },
    progress: { processed: 0, total: 41 },
    result: null,
    ...over,
  };
}
