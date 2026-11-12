/**
 * The editor's zoom and pan: the picture frame is resized and moved (not CSS-scaled), so box
 * borders and handles keep their pixel size at any zoom. Pure maths, so the wheel, the keys and
 * the buttons share one implementation. `x` and `y` are the frame's offset from where the
 * fitted picture sits, in CSS pixels.
 */

export interface Viewport {
  scale: number;
  x: number;
  y: number;
}

export const FIT: Viewport = { scale: 1, x: 0, y: 0 };

/** The widest zoom, as a multiple of the fitted size. */
export const MAX_SCALE = 32;

export interface FrameSize {
  width: number;
  height: number;
}

/** Keeps the picture covering the view: no gap opens beside it, and a fitted picture does not pan. */
export function clampViewport(vp: Viewport, frame: FrameSize): Viewport {
  const scale = Math.min(MAX_SCALE, Math.max(1, vp.scale));
  const minX = frame.width * (1 - scale);
  const minY = frame.height * (1 - scale);
  return {
    scale,
    x: Math.min(0, Math.max(minX, vp.x)),
    y: Math.min(0, Math.max(minY, vp.y)),
  };
}

/** Zooms by `factor` keeping the picture point under `at` where it is. */
export function zoomAt(
  vp: Viewport,
  factor: number,
  at: { x: number; y: number },
  frame: FrameSize,
): Viewport {
  const scale = Math.min(MAX_SCALE, Math.max(1, vp.scale * factor));
  const k = scale / vp.scale;
  return clampViewport(
    { scale, x: at.x - k * (at.x - vp.x), y: at.y - k * (at.y - vp.y) },
    frame,
  );
}

export function panBy(vp: Viewport, dx: number, dy: number, frame: FrameSize): Viewport {
  return clampViewport({ ...vp, x: vp.x + dx, y: vp.y + dy }, frame);
}

/** The zoom at which one picture pixel is one screen pixel, centred on `at`. */
export function actualSize(
  vp: Viewport,
  naturalWidth: number,
  at: { x: number; y: number },
  frame: FrameSize,
): Viewport {
  return zoomAt(vp, naturalWidth / frame.width / vp.scale, at, frame);
}

/** Where the zoomed frame sits in its container, in CSS pixels. */
export function viewportRect(
  vp: Viewport,
  frame: FrameSize,
  container: FrameSize,
): { left: number; top: number; width: number; height: number } {
  return {
    left: (container.width - frame.width) / 2 + vp.x,
    top: (container.height - frame.height) / 2 + vp.y,
    width: frame.width * vp.scale,
    height: frame.height * vp.scale,
  };
}
