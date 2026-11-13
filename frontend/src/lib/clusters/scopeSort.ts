/**
 * The one Sort of an open cluster. Behind it are two controls the cluster view keeps: the order
 * strategy ("outliers first" is one of its orders) and the detector-confidence sort of the
 * facets. The choice maps onto exactly one of them, so the two can never disagree on screen.
 */
export const SCOPE_SORTS = [
  { id: 'default', label: 'Cluster order' },
  { id: 'outliers', label: 'Outliers first' },
  { id: 'confidence_desc', label: 'Detector confidence, high first' },
  { id: 'confidence_asc', label: 'Detector confidence, low first' },
] as const;
export type ScopeSortId = (typeof SCOPE_SORTS)[number]['id'];

export function scopeSortOf(
  outliersFirst: boolean,
  detectorSort: 'default' | 'desc' | 'asc',
): ScopeSortId {
  if (detectorSort === 'desc') return 'confidence_desc';
  if (detectorSort === 'asc') return 'confidence_asc';
  return outliersFirst ? 'outliers' : 'default';
}

export interface ScopeSortTarget {
  /** The order strategy's sort id (`default`, `outliers`, ...). */
  setOrder: (id: 'default' | 'outliers') => void;
  setDetectorSort: (v: 'default' | 'desc' | 'asc') => void;
}

export function applyScopeSort(id: ScopeSortId, to: ScopeSortTarget): void {
  to.setDetectorSort(
    id === 'confidence_desc' ? 'desc' : id === 'confidence_asc' ? 'asc' : 'default',
  );
  to.setOrder(id === 'outliers' ? 'outliers' : 'default');
}
