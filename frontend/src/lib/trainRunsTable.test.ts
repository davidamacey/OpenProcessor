import { describe, expect, it } from 'vitest';
import { bestMapDisplay } from './trainRunsTable';

describe('bestMapDisplay', () => {
  it('shows the served eval.map50 labelled "test" when eval.split is "test"', () => {
    const r = { eval: { split: 'test' as const, map50: 0.917 } };
    expect(bestMapDisplay(r)).toEqual({ value: 0.917, source: 'test' });
  });

  it('shows the served eval.map50 labelled "val" when eval.split is "val" (test pass fell back)', () => {
    const r = { eval: { split: 'val' as const, map50: 0.812 } };
    expect(bestMapDisplay(r)).toEqual({ value: 0.812, source: 'val' });
  });

  it('renders a null source when eval.split is absent (a run predating the eval-split cutover)', () => {
    const r = { eval: { map50: 0.812 } };
    expect(bestMapDisplay(r)).toEqual({ value: 0.812, source: null });
  });

  it('renders null when eval itself is absent', () => {
    expect(bestMapDisplay({ eval: null })).toEqual({ value: null, source: null });
    expect(bestMapDisplay({})).toEqual({ value: null, source: null });
  });

  it('renders null when eval.map50 is null even though split is served', () => {
    const r = { eval: { split: 'test' as const, map50: null } };
    expect(bestMapDisplay(r)).toEqual({ value: null, source: 'test' });
  });
});
