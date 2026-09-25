/**
 * Pure geometry/color helpers for `EmbeddingPlot.svelte` (curation-
 * strategy plan Phase 5, docs/curation-strategy-plan-2026-09.md §2.7/§5.6).
 *
 * Kept out of the component (mirrors the `bboxFrames.ts` /
 * `pager.svelte.ts` split — logic that doesn't need Svelte reactivity
 * lives in a plain, independently-testable module) so the lasso-select
 * math and the data->screen projection can be unit tested without a
 * component-mount harness, which this repo doesn't have.
 *
 * Everything here is pure: no DOM access, no `$state`, no network.
 */

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
    return { toScreen: () => ({ x: cx, y: cy }) };
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
 * Deterministic color for a `cluster_id`. The plot DECORATES an existing
 * assignment (curation-strategy plan §2.7 — "color from cluster_id...
 * decorates, doesn't decide") — never compute/guess a cluster from this
 * color or feed it back into any assignment logic.
 *
 * `null` (no cluster assigned) gets a neutral gray distinct from every
 * palette entry, so "unassigned" reads as a visually distinct bucket
 * rather than colliding with whatever cluster happens to hash to a
 * similar hue.
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
const UNASSIGNED_COLOR = '#71717a'; // zinc-500

export function colorForCluster(clusterId: number | null): string {
  if (clusterId == null || !Number.isFinite(clusterId)) return UNASSIGNED_COLOR;
  const idx = Math.abs(Math.trunc(clusterId)) % PALETTE.length;
  return PALETTE[idx]!;
}

export const PALETTE_SIZE = PALETTE.length;

export interface LegendEntry {
  clusterId: number | null;
  color: string;
  /** Points of this cluster in the served projection. */
  count: number;
  /** The most common served `class_name` among those points, if any. */
  className: string | null;
}

/**
 * F-68 (fresh-start findings 2026-09-25): the plot had no legend, so its
 * colors meant nothing without hovering. The biggest clusters in the
 * served points, each with its color and the class name its points carry
 * most often. Display only: nothing here assigns anything.
 */
export function legendEntries(
  points: readonly { cluster_id: number | null; class_name: string | null }[],
  max = 8,
): LegendEntry[] {
  const byCluster = new Map<
    number | null,
    { count: number; names: Map<string, number> }
  >();
  for (const p of points) {
    let e = byCluster.get(p.cluster_id);
    if (!e) {
      e = { count: 0, names: new Map() };
      byCluster.set(p.cluster_id, e);
    }
    e.count += 1;
    if (p.class_name) e.names.set(p.class_name, (e.names.get(p.class_name) ?? 0) + 1);
  }
  return [...byCluster.entries()]
    .map(([clusterId, e]) => {
      let className: string | null = null;
      let best = 0;
      for (const [n, c] of e.names) {
        if (c > best) {
          best = c;
          className = n;
        }
      }
      return { clusterId, color: colorForCluster(clusterId), count: e.count, className };
    })
    .sort((a, b) => b.count - a.count || (a.clusterId ?? -1) - (b.clusterId ?? -1))
    .slice(0, max);
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
  status: 'idle' | 'running' | 'completed' | 'failed' | 'cancelled',
  wasTracking: boolean,
): RebuildPollOutcome {
  if (status === 'running') return 'running';
  if (!wasTracking) return 'idle';
  if (status === 'completed') return 'completed';
  if (status === 'failed') return 'failed';
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
