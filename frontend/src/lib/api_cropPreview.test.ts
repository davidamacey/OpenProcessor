import { describe, expect, it } from 'vitest';
import { getPreviewUrl } from './api_cropRead';

describe('getPreviewUrl', () => {
  it('asks the preview route for the box alone, enlarged, with no outline', () => {
    const url = new URL(getPreviewUrl('crop 1', 'tight'), 'http://x');
    expect(url.pathname).toMatch(/\/crops\/crop%201\/preview$/);
    expect(url.searchParams.get('pad')).toBe('0');
    expect(url.searchParams.has('outline')).toBe(false);
  });

  it('asks for the box with its surroundings and the box outlined in context mode', () => {
    const url = new URL(getPreviewUrl('c', 'context'), 'http://x');
    expect(url.searchParams.get('pad')).toBe('0.6');
    expect(url.searchParams.get('outline')).toBe('true');
  });
});
