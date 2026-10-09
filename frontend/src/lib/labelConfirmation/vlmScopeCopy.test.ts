import { describe, expect, it } from 'vitest';
import { VLM_SCOPES } from '$lib/types_labelConfirmation';
import { knobsFor, VLM_SCOPE_COPY } from './vlmScopeCopy';

describe('knobsFor', () => {
  it('reads no knob when the VLM is off', () => {
    expect(knobsFor('off')).toEqual([]);
  });

  it('uncertain adds the confidence limit to the common knobs', () => {
    expect(knobsFor('uncertain')).toEqual([
      'conf_max',
      'sample_frac',
      'max_crops_per_day',
    ]);
  });

  it('representatives adds the per-cluster count to the common knobs', () => {
    expect(knobsFor('representatives')).toEqual([
      'per_cluster',
      'sample_frac',
      'max_crops_per_day',
    ]);
  });

  it('all reads only the common knobs', () => {
    expect(knobsFor('all')).toEqual(['sample_frac', 'max_crops_per_day']);
  });
});

describe('VLM_SCOPE_COPY', () => {
  it('does not claim the unlabeled members keep their cluster membership', () => {
    expect(VLM_SCOPE_COPY.representatives.blurb).not.toMatch(/keep their cluster/);
    expect(VLM_SCOPE_COPY.representatives.blurb).toMatch(/moves to its class cluster/);
  });

  it('words every served scope', () => {
    for (const s of VLM_SCOPES) {
      expect(VLM_SCOPE_COPY[s].label.length).toBeGreaterThan(0);
      expect(VLM_SCOPE_COPY[s].blurb.length).toBeGreaterThan(0);
    }
  });
});
