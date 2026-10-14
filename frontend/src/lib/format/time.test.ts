import { describe, expect, it } from 'vitest';
import { formatTime } from './time';

describe('formatTime', () => {
  it('prints the local time of day for an epoch in milliseconds', () => {
    const t = new Date(2026, 8, 24, 21, 13, 41).getTime();
    expect(formatTime(t)).toBe(new Date(t).toLocaleTimeString());
  });

  it('prints the fallback for null, and takes the fallback as a parameter', () => {
    expect(formatTime(null)).toBe('—');
    expect(formatTime(null, '')).toBe('');
  });
});
