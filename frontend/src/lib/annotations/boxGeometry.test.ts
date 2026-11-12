import { describe, expect, it } from 'vitest';
import {
  boxFromPoints,
  clampBox,
  clamp01,
  isDrawable,
  MIN_BOX_SIDE,
  moveBox,
  pointToNorm,
  resizeXyxy,
  toXyxy,
  translateXyxy,
  xyxyFromPoints,
} from './boxGeometry';

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

describe('xyxy helpers', () => {
  it('toXyxy returns the edges of a centre-and-size box', () => {
    const [x1, y1, x2, y2] = toXyxy({ cx: 0.5, cy: 0.5, w: 0.2, h: 0.4 });
    expect([x1, y1, x2, y2].map((n) => +n.toFixed(6))).toEqual([0.4, 0.3, 0.6, 0.7]);
  });

  it('xyxyFromPoints orders and clamps the corners', () => {
    expect(xyxyFromPoints({ x: 0.9, y: 0.2 }, { x: -0.1, y: 0.6 })).toEqual([
      0, 0.2, 0.9, 0.6,
    ]);
  });

  it('a click is not a drawable box and a real drag is', () => {
    expect(isDrawable([0.5, 0.5, 0.5, 0.5])).toBe(false);
    expect(isDrawable([0.5, 0.5, 0.5 + MIN_BOX_SIDE / 2, 0.9])).toBe(false);
    expect(isDrawable([0.1, 0.1, 0.3, 0.3])).toBe(true);
  });

  it('translateXyxy slides to the edge without shrinking', () => {
    const [x1, y1, x2, y2] = translateXyxy([0.6, 0.1, 0.9, 0.3], 0.5, -0.5);
    expect([x1, y1, x2, y2].map((n) => +n.toFixed(6))).toEqual([0.7, 0, 1, 0.2]);
  });

  it('resizeXyxy moves only the edges the handle owns', () => {
    expect(resizeXyxy([0.2, 0.2, 0.6, 0.6], 'se', 0.1, 0.2)).toEqual([
      0.2, 0.2, 0.7, 0.8,
    ]);
    expect(resizeXyxy([0.2, 0.2, 0.6, 0.6], 'n', 0.3, -0.1)).toEqual([
      0.2, 0.1, 0.6, 0.6,
    ]);
  });

  it('resizeXyxy stops at the minimum side instead of flipping', () => {
    const [x1, , x2] = resizeXyxy([0.2, 0.2, 0.6, 0.6], 'e', -0.9, 0);
    expect(x1).toBe(0.2);
    expect(x2).toBeCloseTo(0.2 + MIN_BOX_SIDE);
  });

  it('resizeXyxy clamps to the frame', () => {
    expect(resizeXyxy([0.2, 0.2, 0.6, 0.6], 'nw', -0.9, -0.9)).toEqual([0, 0, 0.6, 0.6]);
  });
});
