/**
 * The `/clusters` list's sort and unlabeled-only choice survive a trip into
 * a cluster and back (sessionStorage), per project. Whatever is stored is
 * untrusted: a stale or hand-edited `sort` falls back to the page default.
 */
import type { ClusterFilter } from '$lib/types';

type ClusterSort = NonNullable<ClusterFilter['sort']>;

const CLUSTER_SORTS: readonly ClusterSort[] = [
  'purity_asc',
  'purity_desc',
  'size_desc',
  'size_asc',
  'dominant_class',
];

export interface PersistedClusterFilter {
  sort: ClusterSort | null;
  unlabeledOnly: boolean;
}

export function filterPersistKey(slug: string): string {
  return `clusters_filter_v1:${slug}`;
}

export function parsePersistedFilter(raw: string | null): PersistedClusterFilter {
  const none: PersistedClusterFilter = { sort: null, unlabeledOnly: false };
  if (raw == null) return none;
  let v: unknown;
  try {
    v = JSON.parse(raw);
  } catch {
    return none;
  }
  if (v == null || typeof v !== 'object') return none;
  const o = v as Record<string, unknown>;
  return {
    sort: CLUSTER_SORTS.includes(o.sort as ClusterSort) ? (o.sort as ClusterSort) : null,
    unlabeledOnly: o.unlabeledOnly === true,
  };
}
