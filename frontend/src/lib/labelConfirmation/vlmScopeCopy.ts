/**
 * Wording and field gating for the VLM scope panel (#119). The scope ids
 * are the contract enum; the sentences are the plan's semantics
 * (`docs/design/label_confirmation_plan.md` section 4). Which knob a scope
 * reads is the plan's definition of that scope, not a client decision.
 */
import type { VlmScope } from '$lib/types_labelConfirmation';

export const VLM_SCOPE_COPY: Record<VlmScope, { label: string; blurb: string }> = {
  all: {
    label: 'Everything',
    blurb:
      'The VLM labels every unlabeled, embedded crop that no human has validated. This is the default.',
  },
  uncertain: {
    label: 'Uncertain only',
    blurb:
      'Only unlabeled crops, crops whose detector confidence is below the limit, low-confidence VLM answers, and crops whose class disagrees with their cluster.',
  },
  representatives: {
    label: 'Cluster representatives',
    blurb:
      'Only the closest members of each cluster, plus crops with no cluster. Far fewer calls; the rest keep their cluster membership.',
  },
  off: {
    label: 'Off',
    blurb:
      'The worker and auto-label runs write no VLM classes. Explicit requests for one crop or one cluster still work.',
  },
};

export type VlmKnob = 'conf_max' | 'per_cluster' | 'sample_frac' | 'max_crops_per_day';

/** The knobs a scope reads; `off` reads none. */
export function knobsFor(scope: VlmScope): VlmKnob[] {
  if (scope === 'off') return [];
  const common: VlmKnob[] = ['sample_frac', 'max_crops_per_day'];
  if (scope === 'uncertain') return ['conf_max', ...common];
  if (scope === 'representatives') return ['per_cluster', ...common];
  return common;
}

export const VLM_POLICY_EFFECT =
  "Takes effect on the VLM worker's next poll (within about 30 seconds) and on the next auto-label run. Stored labels are untouched. A VLM label is a suggestion until a human validates it, whatever the scope.";
