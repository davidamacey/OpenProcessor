import { describe, expect, it } from 'vitest';
import { pickerRecent } from './pickerRecent';

describe('pickerRecent', () => {
  it('puts the proposal first and does not repeat it', () => {
    expect(pickerRecent('bus', ['car', 'bus'])).toEqual(['bus', 'car']);
  });

  it('is just the hints when nothing is proposed', () => {
    expect(pickerRecent(null, ['car'])).toEqual(['car']);
    expect(pickerRecent('', [])).toEqual([]);
  });
});
