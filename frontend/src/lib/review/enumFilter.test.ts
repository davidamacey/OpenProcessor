import { describe, expect, it } from 'vitest';
import { enumFilterSelection, enumServedDefault } from './enumFilter';

const spec = {
  options: [
    { value: 'true', label: 'Only negative frames' },
    { value: 'false', label: 'No negative frames' },
  ],
};

describe('enumFilterSelection', () => {
  it('is empty ("any") when nothing is chosen and no default is served: never the first option', () => {
    expect(enumFilterSelection(spec, undefined, null)).toBe('');
    expect(enumFilterSelection(spec, undefined, 'gone')).toBe('');
  });
  it('prefers the operator pick, then the spec default, then the tab default', () => {
    expect(enumFilterSelection(spec, 'false', 'true')).toBe('false');
    expect(enumFilterSelection(spec, undefined, 'false')).toBe('false');
    expect(enumFilterSelection({ ...spec, default: 'true' }, undefined, 'false')).toBe(
      'true',
    );
  });
  it('a served null spec default and no tab default is "any"', () => {
    expect(enumFilterSelection({ ...spec, default: null }, undefined, null)).toBe('');
  });
  it('enumServedDefault ignores a default that is not one of the options', () => {
    expect(enumServedDefault({ ...spec, default: 'gone' }, undefined)).toBeNull();
    expect(enumServedDefault({ ...spec, default: 'false' }, undefined)).toBe('false');
  });
});
