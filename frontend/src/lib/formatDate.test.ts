import { describe, expect, it } from 'vitest';
import { formatDateOnly, formatTimestamp } from './formatDate';

describe('formatTimestamp (visual audit K7)', () => {
  it('drops microseconds and the offset suffix, normalised to UTC', () => {
    expect(formatTimestamp('2026-09-24T21:13:41.995236+00:00')).toBe(
      '2026-09-24 21:13:41 UTC',
    );
    expect(formatTimestamp('2026-09-24T23:13:41-02:00')).toBe('2026-09-25 01:13:41 UTC');
  });

  it('renders a dash for a missing value and the raw text when unparseable', () => {
    expect(formatTimestamp(null)).toBe('—');
    expect(formatTimestamp('yesterday')).toBe('yesterday');
  });
});

describe('formatDateOnly (m14, 2026-09-24 interactive pass)', () => {
  it('never rolls a date-only string back a day, in any timezone', () => {
    // The old `new Date(x).toLocaleDateString()` bug depended on the
    // runtime's local timezone (regressed only west of UTC); this
    // asserts the exact expected output regardless of TZ, since the fix
    // never constructs a Date from the string at all.
    expect(formatDateOnly('2026-04-29')).toBe('4/29/2026');
  });

  it("does not zero-pad month/day (matches toLocaleDateString's en-US convention)", () => {
    expect(formatDateOnly('2026-01-05')).toBe('1/5/2026');
  });

  it('returns an em dash for null/undefined/empty', () => {
    expect(formatDateOnly(null)).toBe('—');
    expect(formatDateOnly(undefined)).toBe('—');
    expect(formatDateOnly('')).toBe('—');
  });

  it('still handles a full timestamp (defers to Date for anything with a time component)', () => {
    const result = formatDateOnly('2026-04-29T12:00:00Z');
    expect(result).not.toBe('—');
  });

  it('returns an em dash for garbage input', () => {
    expect(formatDateOnly('not-a-date')).toBe('—');
  });
});
