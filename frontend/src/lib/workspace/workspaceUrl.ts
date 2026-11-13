/**
 * The Workspace's URL grammar: the one place that reads or writes the
 * Workspace keys of `page.url.searchParams`. The item-filter keys belong to
 * `ItemFilterState`; this module never touches them.
 */
import { NEAR_DUP_FILTERS, type NearDupFilter } from '$lib/types_images';

const VIEW_IDS = [
  'gallery',
  'clusters',
  'detections',
  'review',
  'table',
  'map',
  'embedding',
] as const;
export type ViewId = (typeof VIEW_IDS)[number];

const SCOPE_KINDS = ['cluster', 'near_dup', 'image'] as const;
const FOCUS_KINDS = ['item', 'image', 'cluster', 'cell'] as const;
const TABLE_ROWS = ['detections', 'images'] as const;
const THUMB_SIZES = ['s', 'm', 'l'] as const;

export type ScopeRef = { kind: (typeof SCOPE_KINDS)[number]; id: string };
export type FocusRef = { kind: (typeof FOCUS_KINDS)[number]; id: string };
export type TableRows = (typeof TABLE_ROWS)[number];
type ThumbSize = (typeof THUMB_SIZES)[number];
type MapPosition = { zoom: number; lat: number; lon: number };

export interface WorkspaceUrlState {
  view: ViewId | null;
  queue: string | null;
  scope: ScopeRef | null;
  focus: FocusRef | null;
  sort: Partial<Record<ViewId, string>>;
  rows: TableRows;
  columns: string[] | null;
  thumb: ThumbSize;
  q: string | null;
  similarTo: string | null;
  /** The Gallery's near-duplicate role (a link from the quality pages opens it collapsed). */
  nearDup: NearDupFilter | null;
  map: MapPosition | null;
}

const sortKey = (view: ViewId): string => `sort_${view}`;

/** Every key this module owns, so a writer can clear stale ones. */
export const WORKSPACE_URL_KEYS: readonly string[] = [
  'view',
  'queue',
  'scope',
  'focus',
  ...VIEW_IDS.map(sortKey),
  'rows',
  'columns',
  'thumb',
  'q',
  'similar_to',
  'near_dup',
  'map',
];

function oneOf<T extends string>(allowed: readonly T[], v: string | null): T | null {
  return v !== null && (allowed as readonly string[]).includes(v) ? (v as T) : null;
}

function ref<K extends string>(
  kinds: readonly K[],
  v: string | null,
): { kind: K; id: string } | null {
  if (!v) return null;
  const at = v.indexOf(':');
  if (at < 1) return null;
  const kind = oneOf(kinds, v.slice(0, at));
  const id = v.slice(at + 1);
  return kind && id ? { kind, id } : null;
}

function parseMap(v: string | null): MapPosition | null {
  const parts = v?.split('/').map(Number);
  if (!parts || parts.length !== 3 || !parts.every(Number.isFinite)) return null;
  const [zoom, lat, lon] = parts;
  return { zoom, lat, lon };
}

export function parseWorkspaceUrl(
  params: Pick<URLSearchParams, 'get'>,
): WorkspaceUrlState {
  const sort: Partial<Record<ViewId, string>> = {};
  for (const view of VIEW_IDS) {
    const v = params.get(sortKey(view));
    if (v) sort[view] = v;
  }
  const columns = params.get('columns');
  return {
    view: oneOf(VIEW_IDS, params.get('view')),
    queue: params.get('queue') || null,
    scope: ref(SCOPE_KINDS, params.get('scope')),
    focus: ref(FOCUS_KINDS, params.get('focus')),
    sort,
    rows: oneOf(TABLE_ROWS, params.get('rows')) ?? 'detections',
    columns: columns ? columns.split(',').filter(Boolean) : null,
    thumb: oneOf(THUMB_SIZES, params.get('thumb')) ?? 'm',
    q: params.get('q') || null,
    similarTo: params.get('similar_to') || null,
    nearDup: oneOf(NEAR_DUP_FILTERS, params.get('near_dup')),
    map: parseMap(params.get('map')),
  };
}

/** Spelled with two decimals for lat and lon, the zoom as given. */
function spellMap(m: MapPosition): string {
  return `${m.zoom}/${m.lat.toFixed(2)}/${m.lon.toFixed(2)}`;
}

function put(params: URLSearchParams, key: string, value: string | null): void {
  if (value) params.set(key, value);
  else params.delete(key);
}

const refText = (r: { kind: string; id: string } | null | undefined): string | null =>
  r ? `${r.kind}:${r.id}` : null;

/** One entry per single-valued state field: its URL key and its spelling. The
 *  defaults (`rows=detections`, `thumb=m`) spell as null, so they are never written. */
const FIELDS: {
  [K in Exclude<keyof WorkspaceUrlState, 'sort'>]: {
    key: string;
    spell: (v: WorkspaceUrlState[K]) => string | null;
  };
} = {
  view: { key: 'view', spell: (v) => v },
  queue: { key: 'queue', spell: (v) => v },
  scope: { key: 'scope', spell: refText },
  focus: { key: 'focus', spell: refText },
  rows: { key: 'rows', spell: (v) => (v === 'images' ? 'images' : null) },
  columns: { key: 'columns', spell: (v) => v?.join(',') ?? null },
  thumb: { key: 'thumb', spell: (v) => (v !== 'm' ? v : null) },
  q: { key: 'q', spell: (v) => v },
  similarTo: { key: 'similar_to', spell: (v) => v },
  nearDup: { key: 'near_dup', spell: (v) => v },
  map: { key: 'map', spell: (v) => (v ? spellMap(v) : null) },
};

/** Writes the keys named in `next`; a key absent from `next` is left alone,
 *  and a null/empty value deletes it. */
export function writeWorkspaceUrl(
  params: URLSearchParams,
  next: Partial<WorkspaceUrlState>,
): void {
  for (const name of Object.keys(FIELDS) as (keyof typeof FIELDS)[]) {
    if (!(name in next)) continue;
    const { key, spell } = FIELDS[name] as {
      key: string;
      spell: (v: unknown) => string | null;
    };
    put(params, key, spell(next[name] ?? null));
  }
  for (const view of VIEW_IDS) {
    if (next.sort && view in next.sort)
      put(params, sortKey(view), next.sort[view] ?? null);
  }
}

/** The query string that opens Review on a served queue, at one item, with any extra queue-only
 *  filter (`combine_conflict=true`). No queue walks the Detections order. */
export function reviewSearch(
  opts: { queue?: string | null; cropId?: string; extra?: Record<string, string> } = {},
): string {
  const params = new URLSearchParams();
  writeWorkspaceUrl(params, {
    view: 'review',
    queue: opts.queue ?? null,
    focus: opts.cropId ? { kind: 'item', id: opts.cropId } : null,
  });
  for (const [key, value] of Object.entries(opts.extra ?? {})) params.set(key, value);
  return `?${params.toString()}`;
}

/** The query string that opens one cluster in the Clusters view. */
export function clusterScopeSearch(uid: string): string {
  const params = new URLSearchParams();
  writeWorkspaceUrl(params, { view: 'clusters', scope: { kind: 'cluster', id: uid } });
  return `?${params.toString()}`;
}
