import { describe, expect, it } from 'vitest';
import { boxFromPoints, clampBox, clamp01, moveBox, pointToNorm } from './boxGeometry';

describe('boxGeometry', () => {
  it('clamp01 pins to the unit interval', () => {
    expect([clamp01(-0.2), clamp01(0.4), clamp01(1.7)]).toEqual([0, 0.4, 1]);
  });

  it('pointToNorm measures against the rect and clamps', () => {
    const rect = { left: 100, top: 50, width: 200, height: 100 };
    expect(pointToNorm(200, 100, rect)).toEqual({ x: 0.5, y: 0.5 });
    expect(pointToNorm(0, 400, rect)).toEqual({ x: 0, y: 1 });
  });

  it('pointToNorm survives a zero-size rect', () => {
    expect(pointToNorm(5, 5, { left: 0, top: 0, width: 0, height: 0 })).toEqual({
      x: 1,
      y: 1,
    });
  });

  it('clampBox cuts a box that leaves the frame', () => {
    const b = clampBox({ cx: 0.9, cy: 0.5, w: 0.4, h: 0.2 });
    expect(b.cx).toBeCloseTo(0.85);
    expect(b.w).toBeCloseTo(0.3);
    expect(b.h).toBeCloseTo(0.2);
  });

  it('boxFromPoints orders the corners', () => {
    const b = boxFromPoints({ x: 0.6, y: 0.8 }, { x: 0.2, y: 0.4 });
    expect(b.cx).toBeCloseTo(0.4);
    expect(b.cy).toBeCloseTo(0.6);
    expect(b.w).toBeCloseTo(0.4);
    expect(b.h).toBeCloseTo(0.4);
  });

  it('moveBox keeps the size and clamps at the edge', () => {
    const b = moveBox({ cx: 0.5, cy: 0.5, w: 0.2, h: 0.2 }, 0.5, 0);
    expect(b.cx).toBeCloseTo(0.95);
    expect(b.w).toBeCloseTo(0.1);
  });
});
