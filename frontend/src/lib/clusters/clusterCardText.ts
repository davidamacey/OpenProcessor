/**
 * Display text for a cluster's two different "purity" numbers
 * (visual audit 2026-09-24, C1/K4). The server serves both:
 *
 * - `purity` — nearest-centroid GEOMETRY purity (share of measured
 *   members whose nearest centroid is this cluster's own).
 * - `label_purity` (mapped onto `Cluster.dominant_pct`) — the dominant
 *   class's share of the LABELLED members.
 *
 * The card subtitle used to print the geometry number as if it were the
 * dominant-class share ("class_b · 3%" on a 616/616 class_b
 * cluster). Every surface now names which one it is showing.
 */
import type { Cluster } from '$lib/types';

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

/** Human name for the served `purity_basis` id. */
export function purityBasisLabel(basis: string | null | undefined): string {
  if (basis == null || basis === 'nearest_centroid') return 'geometry';
  return basis.replace(/_/g, ' ');
}

/** "3% geometry", or null when the cluster has no measured purity. */
export function geometryPurityText(
  c: Pick<Cluster, 'purity' | 'purity_basis'>,
): string | null {
  return c.purity == null ? null : `${pct(c.purity)} ${purityBasisLabel(c.purity_basis)}`;
}
