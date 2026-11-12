/**
 * Pure geometry/color helpers for `EmbeddingPlot.svelte`.
 *
 * Kept out of the component (mirrors the
 * `pager.svelte.ts` split — logic that doesn't need Svelte reactivity
 * lives in a plain, independently-testable module) so the lasso-select
 * math and the data->screen projection can be unit tested without a
 * component-mount harness, which this repo doesn't have.
 *
 * Everything here is pure: no DOM access, no `$state`, no network.
 */

import { jobPhase } from '$lib/jobs/jobPhase';

export interface ScreenPoint {
  x: number;
  y: number;
}

/** A point already projected into canvas pixel space, tagged with the id
 *  the lasso selection reports back (the crop_id). */
export interface ScaledPoint extends ScreenPoint {
  id: string;
}

export interface PlotScale {
  /** Project a data-space (x, y) into canvas pixel space. */
  toScreen(x: number, y: number): ScreenPoint;
  /** The inverse of `toScreen`: a canvas pixel back to data space. */
  toData(x: number, y: number): ScreenPoint;
}

/**
 * Fit every point's bounding box into `[padding, width-padding] x
 * [padding, height-padding]`, preserving neither aspect ratio nor axis
 * meaning (UMAP axes are not directly interpretable) — just a stable,
 * deterministic linear map so re-renders (recoloring, reselecting) don't
 * jitter point positions.
 *
 * Y is flipped (`1 - ...`) so a point with a larger data-space y renders
 * higher on screen — canvas pixel-y grows downward, and an unflipped map
 * would look upside-down relative to how every other chart in this app's
 * design language (and any external tool the operator might cross-check
 * against) orients a scatter plot.
 *
 * Degenerate inputs (no points, zero-size canvas, all points coincident)
 * degrade to a single fixed point/no-op rather than dividing by zero or
 * throwing — a canvas with one dot in the middle is a valid, harmless
 * render of "everything projected to the same place."
 */
export function computeScale(
  points: ReadonlyArray<{ x: number; y: number }>,
  width: number,
  height: number,
  padding = 16,
): PlotScale {
  if (points.length === 0 || width <= 0 || height <= 0) {
    const cx = width / 2;
    const cy = height / 2;
    return { toScreen: () => ({ x: cx, y: cy }), toData: () => ({ x: 0, y: 0 }) };
  }
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const innerW = Math.max(1, width - padding * 2);
  const innerH = Math.max(1, height - padding * 2);
  return {
    toScreen(x: number, y: number): ScreenPoint {
      return {
        x: padding + ((x - minX) / spanX) * innerW,
        y: padding + (1 - (y - minY) / spanY) * innerH,
      };
    },
    toData(x: number, y: number): ScreenPoint {
      return {
        x: minX + ((x - padding) / innerW) * spanX,
        y: minY + (1 - (y - padding) / innerH) * spanY,
      };
    },
  };
}

/**
 * Standard ray-casting point-in-polygon test. `polygon` is an ordered
 * (not necessarily closed) list of screen-space vertices — the lasso
 * path the operator dragged. Fewer than 3 vertices can't enclose
 * anything.
 */
export function pointInPolygon(
  pt: ScreenPoint,
  polygon: ReadonlyArray<ScreenPoint>,
): boolean {
  if (polygon.length < 3) return false;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const pi = polygon[i]!;
    const pj = polygon[j]!;
    const intersect =
      pi.y > pt.y !== pj.y > pt.y &&
      pt.x < ((pj.x - pi.x) * (pt.y - pi.y)) / (pj.y - pi.y) + pi.x;
    if (intersect) inside = !inside;
  }
  return inside;
}

/**
 * The lasso-select-to-crop-ids logic itself: given every point already
 * projected to screen space (tagged with its crop_id) and the lasso
 * path the operator dragged, return the ids that fall inside it.
 *
 * This is the ONLY thing `EmbeddingPlot.svelte` needs to turn a mouse
 * drag into a set of crop ids — what happens with those ids (bulkLabel,
 * moveCropsToCluster) is entirely the component's concern, kept out of
 * this pure module on purpose.
 */
export function selectIdsInLasso(
  points: ReadonlyArray<ScaledPoint>,
  polygon: ReadonlyArray<ScreenPoint>,
): string[] {
  if (polygon.length < 3) return [];
  const out: string[] = [];
  for (const p of points) {
    if (pointInPolygon({ x: p.x, y: p.y }, polygon)) out.push(p.id);
  }
  return out;
}

/**
 * The palette a served legend slot (`colour_index`) maps to. The server decides which class
 * has which slot; the plot only turns a slot into a colour (slots past the palette wrap).
 */
const PALETTE: readonly string[] = [
  '#60a5fa', // blue-400
  '#f472b6', // pink-400
  '#34d399', // emerald-400
  '#fbbf24', // amber-400
  '#a78bfa', // violet-400
  '#f87171', // red-400
  '#22d3ee', // cyan-400
  '#facc15', // yellow-400
  '#4ade80', // green-400
  '#fb923c', // orange-400
  '#e879f9', // fuchsia-400
  '#38bdf8', // sky-400
];

export function colourForIndex(index: number): string {
  return PALETTE[((index % PALETTE.length) + PALETTE.length) % PALETTE.length]!;
}

export const PALETTE_SIZE = PALETTE.length;

/** How a point is painted: dimmed when the filter leaves it out, or when others are selected. */
export function pointAlpha(
  matched: boolean,
  selected: boolean,
  anySelected: boolean,
): number {
  if (!matched) return 0.18;
  return selected || !anySelected ? 0.9 : 0.25;
}

/** The corners of a drag in data space, ordered so that x1 <= x2 and y1 <= y2. */
export function dragToBox(
  scale: PlotScale,
  from: ScreenPoint,
  to: ScreenPoint,
): [number, number, number, number] {
  const a = scale.toData(from.x, from.y);
  const b = scale.toData(to.x, to.y);
  return [Math.min(a.x, b.x), Math.min(a.y, b.y), Math.max(a.x, b.x), Math.max(a.y, b.y)];
}

/** The id of the point nearest `at` within `radius` pixels, or null. */
export function nearestPointId(
  points: ReadonlyArray<ScaledPoint>,
  at: ScreenPoint,
  radius: number,
): string | null {
  let best: string | null = null;
  let bestD = radius * radius;
  for (const p of points) {
    const d = (p.x - at.x) ** 2 + (p.y - at.y) ** 2;
    if (d <= bestD) {
      bestD = d;
      best = p.id;
    }
  }
  return best;
}

/** What the plot should do with one rebuild-job status poll. */
export type RebuildPollOutcome = 'running' | 'completed' | 'failed' | 'idle';

/**
 * Classify a `/viz/projection/status` snapshot. `wasTracking` is whether
 * this component started or adopted the job: a terminal status for a job
 * it never watched (e.g. an old completed run) is just `idle`, so it
 * doesn't toast or reload for someone else's history.
 */
export function classifyRebuildPoll(
  status: string,
  wasTracking: boolean,
): RebuildPollOutcome {
  const phase = jobPhase(status);
  if (phase === 'active') return 'running';
  if (!wasTracking) return 'idle';
  if (phase === 'completed') return 'completed';
  if (phase === 'failed') return 'failed';
  return 'idle';
}

/**
 * m20 (2026-09-24 interactive pass): the plot's point count caption used
 * to always read "{points.length} points", even when `{API_PREFIX}/methods`'
 * `viz_projection` overlay reports a larger `field_coverage_total` — e.g.
 * 116 shown out of a 422-crop pool, no sign anything was left out. Pure
 * so the coverage-notice text is unit-testable without mounting the
 * canvas component.
 */
export function embeddingCoverageSuffix(
  pointsShown: number,
  coveragePoolTotal: number | null,
): string {
  if (coveragePoolTotal != null && coveragePoolTotal > pointsShown) {
    return `of ${coveragePoolTotal.toLocaleString()} projected — Rebuild to include the rest`;
  }
  return 'points';
}
