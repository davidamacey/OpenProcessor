/**
 * Shared bbox-shape plausibility gate.
 *
 * Mirrors the server-side `is_plausible_plate_bbox` in
 * openprocessor:src/services/legacy/plate_detect.py. Defense in depth —
 * flags rows whose stored child bbox is implausible *after* projecting
 * into the parent (crop) frame, regardless of whether the server-side
 * gate caught it.
 *
 * Fixes the divergence documented as Finding C.1 in
 * docs/genericization-plan-2026-09-13.md: `api.ts`'s
 * `_platePlausibleEnvelope` guarded on `Number.isFinite` and returned
 * warning=true for non-finite/degenerate input, while
 * `PlateCard.svelte`'s inline `shapeWarning()` had no such guard, so a
 * NaN width/height compared `false` against every bound and silently
 * returned warning=false. A corrupt row rendered a warning in /review
 * but not in /clusters. This module is now the single implementation
 * both call sites use, on RAW xyxy tuples (not the mapped `BBoxNorm`),
 * since `PlateBrowseItem` (from `getPlates`) does not route through
 * `mapRawCrop` and therefore cannot read a pre-computed flag.
 */

export type Xyxy = readonly [number, number, number, number];

export interface ShapeEnvelope {
  /** Minimum width/height ratio. Plates: 1.2 (wide). */
  aspectMin?: number;
  /** Maximum width/height ratio. Plates: 8.0. */
  aspectMax?: number;
  /** Max fraction of parent width the child may span. Plates: 0.5. */
  maxWidthFrac?: number;
  /** Max fraction of parent height the child may span. */
  maxHeightFrac?: number;
  /** Max fraction of parent AREA (w·h in parent frame). Plates: 0.15. */
  maxAreaFrac?: number;
}

/** The envelope license_plate has always used (aspect 1.2-8.0, width<=0.5, area<=0.15). */
export const PLATE_SHAPE_ENVELOPE: ShapeEnvelope = {
  aspectMin: 1.2,
  aspectMax: 8.0,
  maxWidthFrac: 0.5,
  maxAreaFrac: 0.15,
};

/**
 * Project `childXyxy` into `parentXyxy`'s frame and evaluate `envelope`.
 *
 * Returns `true` (warn) for any non-finite or degenerate input — a
 * corrupt/missing box is treated as implausible, not silently passed.
 * This is the behavior `api.ts` already had; `PlateCard.svelte` is the
 * one that changes (see Finding C.1 / CHANGELOG).
 */
export function evaluateShapeGate(
  childXyxy: Xyxy | number[] | null | undefined,
  parentXyxy: Xyxy | number[] | null | undefined,
  envelope: ShapeEnvelope,
): boolean {
  if (!childXyxy || childXyxy.length !== 4) return false;
  if (!parentXyxy || parentXyxy.length !== 4) return false;

  const [vx1 = 0, vy1 = 0, vx2 = 0, vy2 = 0] = parentXyxy;
  const vw = vx2 - vx1;
  const vh = vy2 - vy1;
  if (!Number.isFinite(vw) || !Number.isFinite(vh) || vw <= 1e-9 || vh <= 1e-9) {
    return false;
  }

  const [px1 = 0, py1 = 0, px2 = 0, py2 = 0] = childXyxy;
  const w = (px2 - px1) / vw;
  const h = (py2 - py1) / vh;

  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return true;

  const aspect = w / h;
  if (envelope.aspectMin != null && aspect < envelope.aspectMin) return true;
  if (envelope.aspectMax != null && aspect > envelope.aspectMax) return true;
  if (envelope.maxWidthFrac != null && w > envelope.maxWidthFrac) return true;
  if (envelope.maxHeightFrac != null && h > envelope.maxHeightFrac) return true;
  if (envelope.maxAreaFrac != null && w * h > envelope.maxAreaFrac) return true;
  return false;
}

/** Renders the envelope numbers as the English sentence the ⚠ badge
 *  tooltip shows, so the wording lives in one place instead of being
 *  hand-copied (review/+page.svelte used to hardcode this sentence). */
export function describeEnvelope(envelope: ShapeEnvelope): string {
  const parts: string[] = [];
  if (envelope.aspectMin != null && envelope.aspectMax != null) {
    parts.push(`aspect ∉ [${envelope.aspectMin}, ${envelope.aspectMax}]`);
  }
  if (envelope.maxWidthFrac != null) {
    parts.push(`covers >${Math.round(envelope.maxWidthFrac * 100)}% of parent width`);
  }
  if (envelope.maxHeightFrac != null) {
    parts.push(`covers >${Math.round(envelope.maxHeightFrac * 100)}% of parent height`);
  }
  if (envelope.maxAreaFrac != null) {
    parts.push(`covers >${Math.round(envelope.maxAreaFrac * 100)}% of parent area`);
  }
  return `Bbox shape fails the plausibility envelope (${parts.join(' or ')})`;
}
