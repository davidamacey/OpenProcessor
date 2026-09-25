import { describe, expect, it } from 'vitest';
import { classifyScoresPoll, formatCoverageCounts, formatCoveragePct } from './scores';
import type { ScoreCoverageEntry } from './api';

function entry(overrides: Partial<ScoreCoverageEntry> = {}): ScoreCoverageEntry {
  return {
    field: 'uniqueness_score',
    n_scored: 100,
    total: 7961,
    pct: 1.26,
    ...overrides,
  };
}

describe('formatCoverageCounts', () => {
  it('renders "n_scored / total" with locale separators', () => {
    expect(formatCoverageCounts(entry({ n_scored: 1234, total: 7961 }))).toBe(
      '1,234 / 7,961',
    );
  });

  it('renders "—" when the entry is missing (not "0 / 0")', () => {
    expect(formatCoverageCounts(undefined)).toBe('—');
  });

  it('renders real zeros as "0 / N", not "—" — 0/7961 is data, not absence', () => {
    expect(formatCoverageCounts(entry({ n_scored: 0, total: 7961 }))).toBe('0 / 7,961');
  });
});

describe('formatCoveragePct', () => {
  it('renders "NN%"', () => {
    expect(formatCoveragePct(entry({ pct: 42.5 }))).toBe('42.5%');
  });

  it('renders "—" when the entry is missing', () => {
    expect(formatCoveragePct(undefined)).toBe('—');
  });

  it('renders a real 0% as "0%", not "—"', () => {
    expect(formatCoveragePct(entry({ pct: 0 }))).toBe('0%');
  });
});

describe('classifyScoresPoll', () => {
  it('running stays running', () => {
    expect(classifyScoresPoll('running')).toBe('running');
  });

  it('failed classifies as failed', () => {
    expect(classifyScoresPoll('failed')).toBe('failed');
  });

  it('cancelled classifies as cancelled', () => {
    expect(classifyScoresPoll('cancelled')).toBe('cancelled');
  });

  it('completed classifies as completed', () => {
    expect(classifyScoresPoll('completed')).toBe('completed');
  });

  it('idle (unreachable in practice) still resolves to completed, not left hanging', () => {
    expect(classifyScoresPoll('idle')).toBe('completed');
  });
});
