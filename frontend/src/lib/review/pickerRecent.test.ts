import { describe, expect, it } from 'vitest';
import { pickerRecent } from './pickerRecent';

describe('pickerRecent', () => {
  it('puts the proposal first and does not repeat it', () => {
    expect(pickerRecent('bolt', ['cog', 'bolt'])).toEqual(['bolt', 'cog']);
  });

  it('is just the hints when nothing is proposed', () => {
    expect(pickerRecent(null, ['cog'])).toEqual(['cog']);
    expect(pickerRecent('', [])).toEqual([]);
  });
});
