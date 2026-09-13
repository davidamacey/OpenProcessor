import { describe, it, expect } from 'vitest';
import { computeViewBox } from './viewBox';

describe('computeViewBox', () => {
  it('returns null for a null box', () => {
    expect(computeViewBox(null, 2.5)).toBeNull();
  });

  it('returns null for a degenerate (zero-size) box', () => {
    expect(computeViewBox({ cx: 0.5, cy: 0.5, w: 0, h: 0.1 }, 2.5)).toBeNull();
    expect(computeViewBox({ cx: 0.5, cy: 0.5, w: 0.1, h: 0 }, 2.5)).toBeNull();
  });

  it('expands by padding and squares the viewport off the larger dimension', () => {
    // w=0.1, h=0.04 -> larger dim 0.1 * padding 2.5 = 0.25 square side.
    const vb = computeViewBox({ cx: 0.5, cy: 0.5, w: 0.1, h: 0.04 }, 2.5);
    expect(vb).toEqual({ cx: 0.5, cy: 0.5, w: 0.25, h: 0.25 });
  });

  it('clamps the side to at most 1 (never zooms out past the full crop)', () => {
    const vb = computeViewBox({ cx: 0.5, cy: 0.5, w: 0.9, h: 0.9 }, 2.5);
    expect(vb?.w).toBe(1);
    expect(vb?.h).toBe(1);
  });

  it('clamps the center so the viewport never runs off [0,1]', () => {
    // side=0.25 -> half=0.125; a box near the top-left corner must clamp.
    const vb = computeViewBox({ cx: 0.02, cy: 0.02, w: 0.1, h: 0.1 }, 2.5);
    expect(vb?.cx).toBe(0.125);
    expect(vb?.cy).toBe(0.125);
  });

  it('clamps the center near the bottom-right corner too', () => {
    const vb = computeViewBox({ cx: 0.98, cy: 0.98, w: 0.1, h: 0.1 }, 2.5);
    expect(vb?.cx).toBe(0.875);
    expect(vb?.cy).toBe(0.875);
  });
});
