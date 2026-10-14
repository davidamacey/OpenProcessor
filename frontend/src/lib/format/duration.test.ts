import { describe, expect, it } from 'vitest';
import { formatDuration } from './duration';

describe('formatDuration', () => {
  it('prints seconds, minutes with seconds, and hours with minutes', () => {
    expect(formatDuration(0)).toBe('0s');
    expect(formatDuration(42)).toBe('42s');
    expect(formatDuration(59.6)).toBe('1m 00s');
    expect(formatDuration(183)).toBe('3m 03s');
    expect(formatDuration(3900)).toBe('1h 05m');
    expect(formatDuration(36000 + 59 * 60 + 40)).toBe('10h 59m');
    expect(formatDuration(90000)).toBe('25h 00m');
  });

  it('clamps a negative value to zero', () => {
    expect(formatDuration(-5)).toBe('0s');
  });

  it('prints the fallback for a missing or non-finite value', () => {
    expect(formatDuration(null)).toBe('—');
    expect(formatDuration(undefined)).toBe('—');
    expect(formatDuration(Number.NaN)).toBe('—');
    expect(formatDuration(Number.POSITIVE_INFINITY)).toBe('—');
    expect(formatDuration(null, '--')).toBe('--');
  });
});
