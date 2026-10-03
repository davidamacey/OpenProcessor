import { describe, expect, it } from 'vitest';
import { enumFilterSelection } from './enumFilter';

const spec = {
  options: [
    { value: 'include', label: 'Include negative frames' },
    { value: 'exclude', label: 'Exclude negative frames' },
  ],
};

describe('enumFilterSelection', () => {
  it('falls back to the first served option when nothing matches (no blank select)', () => {
    expect(enumFilterSelection(spec, undefined, null)).toBe('include');
    expect(enumFilterSelection(spec, undefined, 'gone')).toBe('include');
  });
  it('prefers the operator pick, then the served default', () => {
    expect(enumFilterSelection(spec, 'exclude', 'include')).toBe('exclude');
    expect(enumFilterSelection(spec, undefined, 'exclude')).toBe('exclude');
  });
  it('is empty only when the spec has no options', () => {
    expect(enumFilterSelection({ options: [] }, undefined, null)).toBe('');
  });
});
