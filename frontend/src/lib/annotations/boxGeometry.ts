/**
 * Pure box geometry in normalized coordinates: pointer mapping, clamping and the draw and move
 * maths both box editors share. No Svelte, no network, so it is unit-tested without a mount.
 */
import type { BBoxNormLike } from './types';

export function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

/** The rectangle a pointer position is measured against (a `DOMRect` subset). */
export interface FrameRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** A client pixel position as a clamped fraction of `rect`. */
export function pointToNorm(
  clientX: number,
  clientY: number,
  rect: FrameRect,
): { x: number; y: number } {
  return {
    x: clamp01((clientX - rect.left) / (rect.width || 1)),
    y: clamp01((clientY - rect.top) / (rect.height || 1)),
  };
}

/** Clamps every edge of `b` into the unit square and recomputes centre and size. */
export function clampBox(b: BBoxNormLike): BBoxNormLike {
  const x1 = clamp01(b.cx - b.w / 2);
  const y1 = clamp01(b.cy - b.h / 2);
  const x2 = clamp01(b.cx + b.w / 2);
  const y2 = clamp01(b.cy + b.h / 2);
  return {
    cx: (x1 + x2) / 2,
    cy: (y1 + y2) / 2,
    w: Math.max(0, x2 - x1),
    h: Math.max(0, y2 - y1),
  };
}

/** The box spanned by two drag points, in either order, clamped. */
export function boxFromPoints(
  a: { x: number; y: number },
  b: { x: number; y: number },
): BBoxNormLike {
  const x1 = Math.min(a.x, b.x);
  const y1 = Math.min(a.y, b.y);
  const x2 = Math.max(a.x, b.x);
  const y2 = Math.max(a.y, b.y);
  return clampBox({ cx: (x1 + x2) / 2, cy: (y1 + y2) / 2, w: x2 - x1, h: y2 - y1 });
}

/** `initial` moved by the drag offset, clamped. */
export function moveBox(initial: BBoxNormLike, dx: number, dy: number): BBoxNormLike {
  return clampBox({
    cx: initial.cx + dx,
    cy: initial.cy + dy,
    w: initial.w,
    h: initial.h,
  });
}
