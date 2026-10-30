/**
 * Display text for a cluster's two different "purity" numbers
 * (visual audit 2026-09-24, C1/K4). The server serves both:
 *
 * - `purity` — nearest-centroid GEOMETRY purity (share of measured
 *   members whose nearest centroid is this cluster's own).
 * - `label_agreement` (mapped onto `Cluster.dominant_pct`) — the dominant
 *   class's share of the LABELLED members.
 *
 * The card subtitle used to print the geometry number as if it were the
 * dominant-class share ("class_b · 3%" on a 616/616 class_b
 * cluster). Every surface now names which one it is showing.
 */
import type { Cluster } from '$lib/types';
import type { ValidationState } from '$lib/types_clusterList';

function pct(v: number): string {
  return `${Math.round(v * 100)}%`;
}

/** "100% of labeled", or null when the server sent no label share. */
export function dominantShareText(c: Pick<Cluster, 'dominant_pct'>): string | null {
  return c.dominant_pct == null ? null : `${pct(c.dominant_pct)} of labeled`;
}

/** Tooltip for the dominant-share figure: the served counts behind it. */
export function dominantShareTitle(
  c: Pick<Cluster, 'dominant_count' | 'labelled_count' | 'size' | 'dominant_class_name'>,
): string {
  const name = c.dominant_class_name ?? 'dominant class';
  if (c.dominant_count != null && c.labelled_count != null) {
    return `${c.dominant_count} of ${c.labelled_count} labeled members are ${name} (cluster size ${c.size})`;
  }
  return `Share of labeled members that are ${name}`;
}

/**
 * "purity 19% · noisy" on a class
 * cluster whose labels are 90% right read as "bad labels". The served
 * `purity` measures geometry, so the UI calls it cohesion. The value is
 * still the served `purity`/`purity_n`; the tier word stays the served
 * band.
 */
export const COHESION_TOOLTIP =
  'Cohesion: share of measured members whose nearest cluster centre is this one. It measures how tight the cluster is in embedding space, not whether its labels are right.';

/** "cohesion 19% · n=984", "cohesion 19%", or null with no served value. */
export function cohesionText(c: Pick<Cluster, 'purity' | 'purity_n'>): string | null {
  if (c.purity == null) return null;
  const base = `cohesion ${pct(c.purity)}`;
  return c.purity_n != null ? `${base} · n=${c.purity_n}` : base;
}

/** "widget_x 62% · conf 0.81", or null when the detector served no class. */
export function detectorText(
  c: Pick<Cluster, 'detector_top_class' | 'detector_top_share' | 'detector_conf_mean'>,
): string | null {
  if (!c.detector_top_class) return null;
  const share = c.detector_top_share == null ? '' : ` ${pct(c.detector_top_share)}`;
  const conf =
    c.detector_conf_mean == null ? '' : ` · conf ${c.detector_conf_mean.toFixed(2)}`;
  return `${c.detector_top_class}${share}${conf}`;
}

export const DETECTOR_TOOLTIP =
  "The detector's most common class among the members, its share of the members that have one, and the mean detector confidence.";

/** The words for the three served validation states. */
export const VALIDATION_TEXT: Record<ValidationState, string> = {
  none: 'not validated',
  partial: 'partly validated',
  complete: 'validated',
};

/** "all: 7 · outliers: 3": the served per-tab review counts. */
export function reviewCountsTitle(counts: Record<string, number> | undefined): string {
  const parts = Object.entries(counts ?? {}).map(([tab, n]) => `${tab}: ${n}`);
  return parts.length > 0 ? `Members in each review tab. ${parts.join(' · ')}` : '';
}
