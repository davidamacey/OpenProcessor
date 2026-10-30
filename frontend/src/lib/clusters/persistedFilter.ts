/**
 * The /clusters list's sort and card filter survive a trip into a cluster
 * and back (sessionStorage), per project. Whatever is stored is untrusted:
 * a stale or hand-edited value falls back to the default.
 */
import {
  EMPTY_CLUSTER_FILTER,
  asSize,
  asSortId,
  asStrings,
  asValidationState,
} from '$lib/clusters/clusterListFilter';
import type { ClusterListFilter, ClusterSortId } from '$lib/types_clusterList';

export interface PersistedClusterFilter {
  sort: ClusterSortId | null;
  filter: ClusterListFilter;
}

export function filterPersistKey(slug: string): string {
  return `clusters_filter_v2:${slug}`;
}

export function parsePersistedFilter(raw: string | null): PersistedClusterFilter {
  const none: PersistedClusterFilter = { sort: null, filter: EMPTY_CLUSTER_FILTER };
  if (raw == null) return none;
  let v: unknown;
  try {
    v = JSON.parse(raw);
  } catch {
    return none;
  }
  if (v == null || typeof v !== 'object') return none;
  const o = v as Record<string, unknown>;
  const f = (o.filter != null && typeof o.filter === 'object' ? o.filter : {}) as Record<
    string,
    unknown
  >;
  return {
    sort: asSortId(o.sort),
    filter: {
      dominant_class: asStrings(f.dominant_class),
      detector_class: asStrings(f.detector_class),
      min_size: asSize(f.min_size),
      max_size: asSize(f.max_size),
      validation_state: asValidationState(f.validation_state),
      flag: asStrings(f.flag),
      needs_review: f.needs_review === true,
    },
  };
}
