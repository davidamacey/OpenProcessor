import { describe, expect, it } from 'vitest';
import { enumFilterSelection } from './enumFilter';

const withAny = {
  default: null,
  options: [
    { value: '', label: 'Any' },
    { value: 'include', label: 'Include negative frames' },
    { value: 'exclude', label: 'Exclude negative frames' },
  ],
};
const noAny = {
  default: 'all',
  options: [
    { value: 'all', label: 'All' },
    { value: 'detected', label: 'Detected only' },
  ],
};

describe('enumFilterSelection', () => {
  it('shows the served Any option when nothing is chosen and the default is null', () => {
    expect(enumFilterSelection(withAny, undefined)).toBe('');
  });
  it('shows the served default when nothing is chosen', () => {
    expect(enumFilterSelection({ ...noAny, default: 'detected' }, undefined)).toBe(
      'detected',
    );
    expect(enumFilterSelection(noAny, undefined)).toBe('all');
  });
  it('keeps an explicit Any pick instead of the default', () => {
    expect(enumFilterSelection(withAny, '')).toBe('');
    expect(enumFilterSelection({ ...withAny, default: 'include' }, '')).toBe('');
  });
  it('prefers the operator pick', () => {
    expect(enumFilterSelection(withAny, 'exclude')).toBe('exclude');
  });
  it('never stands in the first option for an unset value', () => {
    expect(
      enumFilterSelection({ default: null, options: noAny.options }, undefined),
    ).toBe('');
    expect(
      enumFilterSelection({ default: 'gone', options: noAny.options }, 'also-gone'),
    ).toBe('');
  });
  it('an empty pick falls back to the default when the spec offers no Any option', () => {
    expect(enumFilterSelection(noAny, '')).toBe('all');
  });
});
