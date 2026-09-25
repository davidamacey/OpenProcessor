import { describe, expect, it } from 'vitest';
import { bestMapDisplay } from './trainRunsTable';

describe('bestMapDisplay', () => {
  it('shows the served test-split eval.map50 when eval.split is "test"', () => {
    const r = {
      eval: { split: 'test' as const, map50: 0.917 },
      best_metric: { map50: 0.995 },
    };
    expect(bestMapDisplay(r)).toEqual({ value: 0.917, source: 'test' });
  });

  it('falls back to best_metric.map50 when eval.split is "val"', () => {
    const r = {
      eval: { split: 'val' as const, map50: 0.812 },
      best_metric: { map50: 0.995 },
    };
    expect(bestMapDisplay(r)).toEqual({ value: 0.995, source: 'val' });
  });

  it('falls back to best_metric.map50 when eval.split is absent (pre-e9aac68 backend)', () => {
    const r = { eval: { map50: 0.812 }, best_metric: { map50: 0.995 } };
    expect(bestMapDisplay(r)).toEqual({ value: 0.995, source: 'val' });
  });

  it('falls back to best_metric.map50 when eval.split is "test" but eval.map50 is null', () => {
    const r = {
      eval: { split: 'test' as const, map50: null },
      best_metric: { map50: 0.995 },
    };
    expect(bestMapDisplay(r)).toEqual({ value: 0.995, source: 'val' });
  });

  it('falls back to null when neither eval nor best_metric carry a value', () => {
    const r = { eval: null, best_metric: null };
    expect(bestMapDisplay(r)).toEqual({ value: null, source: 'val' });
  });
});
