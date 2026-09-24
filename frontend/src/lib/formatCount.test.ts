import { describe, expect, it } from 'vitest';
import { formatCount } from './formatCount';

describe('formatCount', () => {
  it('renders null as an em dash, not 0', () => {
    expect(formatCount(null)).toBe('—');
  });

  it('renders undefined as an em dash', () => {
    expect(formatCount(undefined)).toBe('—');
  });

  it('renders 0 as "0", not an em dash — a real zero is not unknown', () => {
    expect(formatCount(0)).toBe('0');
  });

  it('formats a real count with locale grouping', () => {
    expect(formatCount(1234)).toBe('1,234');
  });
});
