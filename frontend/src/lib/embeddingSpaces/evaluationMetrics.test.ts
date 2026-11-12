import { describe, expect, it } from 'vitest';
import { metricRows } from './evaluationMetrics';

describe('metricRows', () => {
  it('lists every leaf with a dotted path, in the served order', () => {
    expect(
      metricRows({
        recall_at_10: { baseline: 0.71234, candidate: 0.78 },
        n: 2000,
        strata: ['a', 'b'],
        note: null,
      }),
    ).toEqual([
      { path: 'recall_at_10.baseline', value: '0.7123' },
      { path: 'recall_at_10.candidate', value: '0.78' },
      { path: 'n', value: '2000' },
      { path: 'strata', value: 'a, b' },
      { path: 'note', value: 'null' },
    ]);
  });

  it('is empty for an empty document', () => {
    expect(metricRows({})).toEqual([]);
  });
});
