/**
 * Pure math behind the frozen-viewport bbox-editor zoom, extracted from
 * `_seedViewBox()` in `review/+page.svelte` (Phase 0 seam,
 * docs/genericization-plan-2026-09-13.md §3.5 point 1).
 *
 * The `untrack()` wrapping that makes this "frozen" (recomputed only
 * when the crop changes, not on every drag tick) is a Svelte-reactivity
 * concern that has to stay in the component — there is nothing to
 * extract there. What IS extractable, and was previously untested, is
 * the padding/squaring/clamping math itself.
 */

import type { BBoxNorm } from '../types';

/**
 * Expands `box` by `padding`, squares the viewport (the canvas is
 * aspect-square; a non-square viewBox would re-introduce letterboxing),
 * and clamps the center so the viewport never runs off the [0,1] crop
 * frame. Returns `null` for a degenerate or missing box (nothing to
 * zoom in on).
 */
export function computeViewBox(box: BBoxNorm | null, padding: number): BBoxNorm | null {
  if (!box) return null;
  const w0 = box.w;
  const h0 = box.h;
  if (w0 <= 0 || h0 <= 0) return null;

  const side = Math.min(1, Math.max(w0, h0) * padding);
  const half = side / 2;
  const cx = Math.min(1 - half, Math.max(half, box.cx));
  const cy = Math.min(1 - half, Math.max(half, box.cy));
  return { cx, cy, w: side, h: side };
}
