import { describe, expect, it } from 'vitest';
import { mixedBreakdown } from './mixedBreakdown';

describe('mixedBreakdown', () => {
  it('lists each detector class with its served count', () => {
    expect(
      mixedBreakdown([
        { value: 'person', count: 19 },
        { value: 'backpack', count: 5 },
      ]),
    ).toBe('Mixed: 19 person, 5 backpack.');
  });
  it('is null for one class or none', () => {
    expect(mixedBreakdown([{ value: 'person', count: 24 }])).toBeNull();
    expect(mixedBreakdown([])).toBeNull();
  });
});
