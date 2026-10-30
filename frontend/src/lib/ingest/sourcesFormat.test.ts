import { describe, expect, it } from 'vitest';
import { formatWhen } from './sourcesFormat';

describe('formatWhen', () => {
  it('writes an ISO string in the shared timestamp format', () => {
    expect(formatWhen('2026-10-01T12:34:56.789+00:00')).toBe('2026-10-01 12:34:56 UTC');
  });

  it('writes epoch seconds the same way', () => {
    expect(formatWhen(Date.UTC(2026, 9, 1, 12, 34, 56) / 1000)).toBe(
      '2026-10-01 12:34:56 UTC',
    );
  });

  it('keeps the dash for an absent or unparseable value', () => {
    expect(formatWhen(null)).toBe('--');
    expect(formatWhen('')).toBe('--');
    expect(formatWhen('not a date')).toBe('--');
  });
});
