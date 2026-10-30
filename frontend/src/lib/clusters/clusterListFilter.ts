/**
 * The /clusters card filter: the empty value, the query parameters it
 * sends, and its URL form. The server decides what each field matches;
 * nothing here ranks, counts or narrows a list.
 */
import {
  CLUSTER_SORT_IDS,
  VALIDATION_STATES,
  type ClusterListFilter,
  type ClusterSortId,
  type ValidationState,
} from '$lib/types_clusterList';

export const EMPTY_CLUSTER_FILTER: ClusterListFilter = {
  dominant_class: [],
  detector_class: [],
  min_size: null,
  max_size: null,
  validation_state: null,
  flag: [],
  needs_review: false,
};

/** Every URL key this filter owns (the sort included). */
export const CLUSTER_LIST_URL_KEYS = [
  'sort',
  'dominant_class',
  'detector_class',
  'min_size',
  'max_size',
  'validation_state',
  'flag',
  'needs_review',
] as const;

export function isClusterFilterActive(f: ClusterListFilter): boolean {
  return (
    f.dominant_class.length > 0 ||
    f.detector_class.length > 0 ||
    f.min_size != null ||
    f.max_size != null ||
    f.validation_state != null ||
    f.flag.length > 0 ||
    f.needs_review
  );
}

/** The set fields as `GET /clusters` query parameters (unset fields omitted). */
export function clusterFilterToQuery(f: ClusterListFilter): Partial<ClusterListFilter> {
  const q: Partial<ClusterListFilter> = {};
  if (f.dominant_class.length > 0) q.dominant_class = f.dominant_class;
  if (f.detector_class.length > 0) q.detector_class = f.detector_class;
  if (f.min_size != null) q.min_size = f.min_size;
  if (f.max_size != null) q.max_size = f.max_size;
  if (f.validation_state != null) q.validation_state = f.validation_state;
  if (f.flag.length > 0) q.flag = f.flag;
  if (f.needs_review) q.needs_review = true;
  return q;
}

export function asSortId(v: unknown): ClusterSortId | null {
  return CLUSTER_SORT_IDS.includes(v as ClusterSortId) ? (v as ClusterSortId) : null;
}

export function asValidationState(v: unknown): ValidationState | null {
  return VALIDATION_STATES.includes(v as ValidationState) ? (v as ValidationState) : null;
}

export function asSize(v: unknown): number | null {
  return typeof v === 'number' && Number.isInteger(v) && v >= 0 ? v : null;
}

export function asStrings(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
}

function uniq(values: string[]): string[] {
  return values.filter((v, i) => v !== '' && values.indexOf(v) === i);
}

/** Writes `sort` and the filter into `params`, deleting keys that are now empty. */
export function clusterFilterToUrl(
  params: URLSearchParams,
  sort: ClusterSortId | null,
  f: ClusterListFilter,
): void {
  for (const k of CLUSTER_LIST_URL_KEYS) params.delete(k);
  if (sort) params.set('sort', sort);
  for (const v of f.dominant_class) params.append('dominant_class', v);
  for (const v of f.detector_class) params.append('detector_class', v);
  if (f.min_size != null) params.set('min_size', String(f.min_size));
  if (f.max_size != null) params.set('max_size', String(f.max_size));
  if (f.validation_state) params.set('validation_state', f.validation_state);
  for (const v of f.flag) params.append('flag', v);
  if (f.needs_review) params.set('needs_review', 'true');
}

/** The sort and filter a URL names, or null when it names none of them. */
export function clusterFilterFromUrl(
  params: Pick<URLSearchParams, 'get' | 'getAll' | 'has'>,
): { sort: ClusterSortId | null; filter: ClusterListFilter } | null {
  if (!CLUSTER_LIST_URL_KEYS.some((k) => params.has(k))) return null;
  const num = (k: string): number | null => {
    const raw = params.get(k);
    return raw == null || raw === '' ? null : asSize(Number(raw));
  };
  return {
    sort: asSortId(params.get('sort')),
    filter: {
      dominant_class: uniq(params.getAll('dominant_class')),
      detector_class: uniq(params.getAll('detector_class')),
      min_size: num('min_size'),
      max_size: num('max_size'),
      validation_state: asValidationState(params.get('validation_state')),
      flag: uniq(params.getAll('flag')),
      needs_review: params.get('needs_review') === 'true',
    },
  };
}
