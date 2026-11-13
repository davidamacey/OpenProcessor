import { describe, expect, it } from 'vitest';
import { plural } from './plural';

describe('plural', () => {
  it('counts with the singular for one and the plural otherwise', () => {
    expect(plural(1, 'item')).toBe('1 item');
    expect(plural(0, 'item')).toBe('0 items');
    expect(plural(24, 'item')).toBe('24 items');
  });
  it('groups thousands and takes an irregular plural', () => {
    expect(plural(1131, 'warning')).toBe('1,131 warnings');
    expect(plural(2, 'class', 'classes')).toBe('2 classes');
    expect(plural(1, 'class', 'classes')).toBe('1 class');
  });
});
