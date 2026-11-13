import { describe, expect, it, vi } from 'vitest';
import { EMPTY_CLUSTER_FILTER } from '$lib/clusters/clusterListFilter';
import { refinementChips, type RefinementState } from './refinementChips';

function state(over: Partial<RefinementState> = {}): RefinementState {
  return {
    filter: { ...EMPTY_CLUSTER_FILTER },
    flagVocabulary: [{ id: 'mixed_detector_classes', label: 'Mixed detector classes' }],
    setFilter: vi.fn(),
    subjectScope: 0,
    setSubjectScope: vi.fn(),
    minBlurRatio: null,
    clearClarity: vi.fn(),
    ...over,
  } as RefinementState;
}

describe('refinementChips', () => {
  it('has no chip when nothing is refined', () => {
    expect(refinementChips(state())).toEqual([]);
  });

  it('names each refinement in force, with the served flag label', () => {
    const chips = refinementChips(
      state({
        filter: {
          ...EMPTY_CLUSTER_FILTER,
          dominant_class: ['cup'],
          min_size: 5,
          max_size: 20,
          validation_state: 'none',
          needs_review: true,
          flag: ['mixed_detector_classes'],
        },
        subjectScope: 1,
        minBlurRatio: 0.4,
      }),
    );
    expect(chips.map((c) => c.label)).toEqual([
      'Class: cup',
      'Size 5 to 20',
      'Validation: not validated',
      'Needs review',
      'Mixed detector classes',
      'Subject: Largest',
      'Minimum clarity set',
    ]);
  });

  it('a chip clears only its own refinement', () => {
    const setFilter = vi.fn();
    const filter = {
      ...EMPTY_CLUSTER_FILTER,
      dominant_class: ['cup', 'bottle'],
      needs_review: true,
    };
    const chips = refinementChips(state({ filter, setFilter }));
    chips.find((c) => c.id === 'dominant:cup')!.clear();
    expect(setFilter).toHaveBeenCalledWith({
      ...filter,
      dominant_class: ['bottle'],
    });
  });
});
