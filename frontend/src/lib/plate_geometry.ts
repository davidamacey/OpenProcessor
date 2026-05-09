/**
 * Plate-bbox coordinate conversion between the source-image frame and the
 * vehicle-crop frame.
 *
 * The backend (openprocessor commit 38b316e) stores `plate_bbox_norm` in the
 * **source-image** frame, while the labeler renders the vehicle crop as a
 * thumbnail. To draw the plate ring on the crop thumbnail (or to let the
 * user edit the plate box on top of the crop), we need to map between the
 * two frames using the parent vehicle's `bbox_norm` (also in source frame).
 *
 *   plate_in_crop_x = (plate.x - vehicle.x1) / (vehicle.x2 - vehicle.x1)
 *
 * Both helpers are pure and clamp the output to `[0, 1]`. The internal
 * representation is the project's center-form `BBoxNorm` ({cx, cy, w, h});
 * conversions to/from the API's [x1, y1, x2, y2] form happen in api.ts.
 */
import type { BBoxNorm } from './types';

/** Tolerable degenerate width below which we refuse the conversion. */
const EPS = 1e-9;

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return 0;
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

interface XYXY {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

function toXYXY(b: BBoxNorm): XYXY {
  const halfW = b.w / 2;
  const halfH = b.h / 2;
  return {
    x1: b.cx - halfW,
    y1: b.cy - halfH,
    x2: b.cx + halfW,
    y2: b.cy + halfH,
  };
}

function fromXYXY(b: XYXY): BBoxNorm {
  // Clamp first, then derive cx/cy/w/h from the clamped corners so the
  // result is guaranteed to fit inside [0, 1] on every component.
  const x1 = clamp01(Math.min(b.x1, b.x2));
  const x2 = clamp01(Math.max(b.x1, b.x2));
  const y1 = clamp01(Math.min(b.y1, b.y2));
  const y2 = clamp01(Math.max(b.y1, b.y2));
  return {
    cx: (x1 + x2) / 2,
    cy: (y1 + y2) / 2,
    w: Math.max(0, x2 - x1),
    h: Math.max(0, y2 - y1),
  };
}

/**
 * Map a plate bbox from the source-image frame into the vehicle-crop
 * frame. Returns null when the parent vehicle box is degenerate (zero
 * width or height) — in that case the divisor would explode.
 */
export function sourceToCropFrame(
  plateBbox: BBoxNorm,
  vehicleBbox: BBoxNorm,
): BBoxNorm | null {
  const v = toXYXY(vehicleBbox);
  const vw = v.x2 - v.x1;
  const vh = v.y2 - v.y1;
  if (vw <= EPS || vh <= EPS) return null;
  const p = toXYXY(plateBbox);
  return fromXYXY({
    x1: (p.x1 - v.x1) / vw,
    y1: (p.y1 - v.y1) / vh,
    x2: (p.x2 - v.x1) / vw,
    y2: (p.y2 - v.y1) / vh,
  });
}

/**
 * Map a plate bbox from the vehicle-crop frame back into the source-image
 * frame. Always returns a value (no degeneracy guard needed since we're
 * multiplying, not dividing); the result is clamped to [0, 1].
 */
export function cropToSourceFrame(
  plateBboxInCrop: BBoxNorm,
  vehicleBbox: BBoxNorm,
): BBoxNorm {
  const v = toXYXY(vehicleBbox);
  const vw = v.x2 - v.x1;
  const vh = v.y2 - v.y1;
  const p = toXYXY(plateBboxInCrop);
  return fromXYXY({
    x1: v.x1 + p.x1 * vw,
    y1: v.y1 + p.y1 * vh,
    x2: v.x1 + p.x2 * vw,
    y2: v.y1 + p.y2 * vh,
  });
}

/** Convert a {cx, cy, w, h} box to the API's [x1, y1, x2, y2] tuple. */
export function bboxNormToXYXY(b: BBoxNorm): [number, number, number, number] {
  const halfW = b.w / 2;
  const halfH = b.h / 2;
  return [
    clamp01(b.cx - halfW),
    clamp01(b.cy - halfH),
    clamp01(b.cx + halfW),
    clamp01(b.cy + halfH),
  ];
}
