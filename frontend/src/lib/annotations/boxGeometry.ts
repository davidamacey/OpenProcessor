/**
 * Pure box geometry in normalized coordinates: pointer mapping, clamping and the draw and move
 * maths both box editors share. No Svelte, no network, so it is unit-tested without a mount.
 */
import type { BBoxNormLike, XYXY } from './types';

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

/** The `[x1, y1, x2, y2]` edges of a centre-and-size box. */
export function toXyxy(b: BBoxNormLike): XYXY {
  return [b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2];
}

/** The smallest box the editor draws or resizes to, as a fraction of the image side. */
export const MIN_BOX_SIDE = 0.004;

/** Whether a drawn box is big enough to keep (a click is not a box). */
export function isDrawable(box: XYXY): boolean {
  return box[2] - box[0] >= MIN_BOX_SIDE && box[3] - box[1] >= MIN_BOX_SIDE;
}

/** The corners of two drag points as ordered, clamped edges. */
export function xyxyFromPoints(
  a: { x: number; y: number },
  b: { x: number; y: number },
): XYXY {
  return [
    clamp01(Math.min(a.x, b.x)),
    clamp01(Math.min(a.y, b.y)),
    clamp01(Math.max(a.x, b.x)),
    clamp01(Math.max(a.y, b.y)),
  ];
}

/** `box` shifted by the offset, keeping its size: it slides to the frame edge, it does not shrink. */
export function translateXyxy(box: XYXY, dx: number, dy: number): XYXY {
  const w = box[2] - box[0];
  const h = box[3] - box[1];
  const x1 = Math.min(1 - w, Math.max(0, box[0] + dx));
  const y1 = Math.min(1 - h, Math.max(0, box[1] + dy));
  return [x1, y1, x1 + w, y1 + h];
}

export type ResizeHandle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';

/** Where each handle sits and which edges it moves. */
export const RESIZE_HANDLES: Record<
  ResizeHandle,
  {
    x: 0 | 0.5 | 1;
    y: 0 | 0.5 | 1;
    west: boolean;
    east: boolean;
    north: boolean;
    south: boolean;
  }
> = {
  nw: { x: 0, y: 0, west: true, east: false, north: true, south: false },
  n: { x: 0.5, y: 0, west: false, east: false, north: true, south: false },
  ne: { x: 1, y: 0, west: false, east: true, north: true, south: false },
  e: { x: 1, y: 0.5, west: false, east: true, north: false, south: false },
  se: { x: 1, y: 1, west: false, east: true, north: false, south: true },
  s: { x: 0.5, y: 1, west: false, east: false, north: false, south: true },
  sw: { x: 0, y: 1, west: true, east: false, north: false, south: true },
  w: { x: 0, y: 0.5, west: true, east: false, north: false, south: false },
};

/** `box` with the edges `handle` owns moved by the offset, clamped to the frame and to the minimum side. */
export function resizeXyxy(
  box: XYXY,
  handle: ResizeHandle,
  dx: number,
  dy: number,
): XYXY {
  const h = RESIZE_HANDLES[handle];
  let [x1, y1, x2, y2] = box;
  if (h.west) x1 = Math.min(x2 - MIN_BOX_SIDE, clamp01(x1 + dx));
  if (h.east) x2 = Math.max(x1 + MIN_BOX_SIDE, clamp01(x2 + dx));
  if (h.north) y1 = Math.min(y2 - MIN_BOX_SIDE, clamp01(y1 + dy));
  if (h.south) y2 = Math.max(y1 + MIN_BOX_SIDE, clamp01(y2 + dy));
  return [x1, y1, x2, y2];
}
