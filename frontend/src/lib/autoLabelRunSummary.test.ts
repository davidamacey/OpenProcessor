import { describe, expect, it } from 'vitest';
import { summarizeRunStages } from './autoLabelRunSummary';

describe('summarizeRunStages (visual audit D5)', () => {
  it('one row per served stage with its status, reason and scalar fields only', () => {
    const rows = summarizeRunStages({
      stages: {
        cluster_id_normalize: { status: 'ok', updated: 0, failures: 0 },
        cluster_residuals: {
          status: 'success',
          method: 'ivf',
          n_clusters: 1,
          cluster_method_params: { niter: 50 },
        },
        auto_promote: { skipped: true, reason: 'disabled by default' },
      },
    });
    expect(rows.map((r) => [r.key, r.status, r.reason])).toEqual([
      ['cluster_id_normalize', 'ok', null],
      ['cluster_residuals', 'success', null],
      ['auto_promote', 'skipped', 'disabled by default'],
    ]);
    expect(rows[1]!.fields).toEqual([
      { name: 'method', value: 'ivf' },
      { name: 'n_clusters', value: '1' },
    ]);
  });

  it('no stages served means no rows', () => {
    expect(summarizeRunStages(null)).toEqual([]);
    expect(summarizeRunStages({})).toEqual([]);
  });
});
