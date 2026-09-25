import { describe, expect, it } from 'vitest';
import {
  evalOverallLabel,
  evalPerClassLabel,
  formatMetric,
  formatScalar,
  isTerminalTrainState,
  metricEpochLabel,
} from './trainResults';
import { trainStatusFixture } from './test/fixtures/trainRun';

describe('evalOverallLabel / evalPerClassLabel', () => {
  it("labels today's split-less eval: overall as validation (last epoch), per-class as test split", () => {
    // trainStatusFixture.eval carries no `split` — the real shape the
    // live backend serves today.
    expect(evalOverallLabel(trainStatusFixture.eval)).toBe('validation (last epoch)');
    expect(evalPerClassLabel(trainStatusFixture.eval)).toBe(
      'test split (frozen holdout)',
    );
  });

  it('labels an upcoming eval with split: "test" as test split for both halves', () => {
    const ev = { ...trainStatusFixture.eval, split: 'test' as const };
    expect(evalOverallLabel(ev)).toBe('test split (frozen holdout)');
    expect(evalPerClassLabel(ev)).toBe('test split (frozen holdout)');
  });

  it('labels an upcoming eval with split: "val" as validation for both halves', () => {
    const ev = { ...trainStatusFixture.eval, split: 'val' as const };
    expect(evalOverallLabel(ev)).toBe('validation');
    expect(evalPerClassLabel(ev)).toBe('validation');
  });

  it('returns empty string for a null/undefined eval', () => {
    expect(evalOverallLabel(null)).toBe('');
    expect(evalPerClassLabel(undefined)).toBe('');
  });
});

describe('formatMetric', () => {
  it('renders null/undefined as an em dash, never 0', () => {
    expect(formatMetric(null)).toBe('—');
    expect(formatMetric(undefined)).toBe('—');
  });

  it('renders a real 0 as "0.000", not "—"', () => {
    expect(formatMetric(0)).toBe('0.000');
  });

  it('formats a real value to 3 decimals by default', () => {
    expect(formatMetric(0.9191)).toBe('0.919');
  });
});

describe('formatScalar', () => {
  it('renders null/undefined/empty-string as an em dash', () => {
    expect(formatScalar(null)).toBe('—');
    expect(formatScalar(undefined)).toBe('—');
    expect(formatScalar('')).toBe('—');
  });

  it('renders a real 0 as "0", not "—"', () => {
    expect(formatScalar(0)).toBe('0');
  });

  it('stringifies a real value verbatim', () => {
    expect(formatScalar('abc123')).toBe('abc123');
    expect(formatScalar(42)).toBe('42');
  });
});

describe('metricEpochLabel', () => {
  it('formats a served epoch number', () => {
    expect(metricEpochLabel({ epoch: 17, map50: 0.9 })).toBe('epoch 17');
  });

  it('returns null when epoch is absent', () => {
    expect(metricEpochLabel({ map50: 0.9 })).toBeNull();
  });

  it('returns null for a null/undefined metric (a run predating this field)', () => {
    expect(metricEpochLabel(null)).toBeNull();
    expect(metricEpochLabel(undefined)).toBeNull();
  });

  it('renders epoch 0 (not falsy-skipped)', () => {
    expect(metricEpochLabel({ epoch: 0, map50: 0.1 })).toBe('epoch 0');
  });
});

describe('isTerminalTrainState', () => {
  it('treats finished/failed/cancelled/skipped/lost as terminal', () => {
    for (const s of ['finished', 'failed', 'cancelled', 'skipped', 'lost']) {
      expect(isTerminalTrainState(s)).toBe(true);
    }
  });

  it('treats queued/starting/running/exporting as non-terminal', () => {
    for (const s of ['queued', 'starting', 'running', 'exporting']) {
      expect(isTerminalTrainState(s)).toBe(false);
    }
  });
});
