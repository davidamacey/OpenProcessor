/**
 * Selection writes (`selection` + `dry_run` on batch label / exclude /
 * unexclude / move), the shared item filter on the list/stats/clusters
 * routes, and `export/yolo item_filter`. Exact request bodies and query
 * keys: a class is sent by NAME (`class_name`, repeated), never `class_id`.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  API_PREFIX,
  batchExcludeSelection,
  batchUnexcludeSelection,
  bulkLabelSelection,
  excludeCrops,
  exportYolo,
  getClusters,
  getMatchingItemCount,
  getRegions,
  moveSelectionToCluster,
  unexcludeCrops,
} from './api';
import type { ItemFilterQuery, ItemSelection } from '$lib/types_itemFilter';

function ok(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

const SEL: ItemSelection = {
  filter: { class_names: ['widget'], origin: ['sam3'] },
  limit: 50,
  sample: 'random',
  seed: 7,
};

function stub(body: unknown) {
  const fetchMock = vi.fn(async (_u: string, _i?: RequestInit) => ok(body));
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

describe('selection writes', () => {
  it('bulkLabelSelection: PUT batch_label with selection, dry_run and no crop_ids', async () => {
    const f = stub({ dry_run: true, selected: 12 });
    const res = await bulkLabelSelection(SEL, 5, true);
    const [url, init] = f.mock.calls[0]!;
    expect(url).toBe(`${API_PREFIX}/crops/batch_label`);
    expect(init?.method).toBe('PUT');
    expect(JSON.parse(String(init?.body))).toEqual({
      selection: SEL,
      class_id: 5,
      validated: true,
      dry_run: true,
    });
    expect(res).toEqual({ dry_run: true, selected: 12 });
  });

  it('batchExcludeSelection: POST batch_exclude with the reason', async () => {
    const f = stub({ excluded: 3, updated_ids: ['a', 'b', 'c'], errors: 0 });
    const res = await batchExcludeSelection(SEL, 'blurry', false);
    const [url, init] = f.mock.calls[0]!;
    expect(url).toBe(`${API_PREFIX}/crops/batch_exclude`);
    expect(init?.method).toBe('POST');
    expect(JSON.parse(String(init?.body))).toEqual({
      selection: SEL,
      reason: 'blurry',
      dry_run: false,
    });
    expect(res).toMatchObject({ excluded: 3, updated_ids: ['a', 'b', 'c'] });
  });

  it('batchUnexcludeSelection asks for excluded items (include_excluded)', async () => {
    const f = stub({ unexcluded: 1, updated_ids: ['a'], errors: 0 });
    await batchUnexcludeSelection(SEL, false);
    const [url, init] = f.mock.calls[0]!;
    expect(url).toBe(`${API_PREFIX}/crops/batch_unexclude`);
    expect(JSON.parse(String(init?.body))).toEqual({
      selection: { ...SEL, include_excluded: true },
      dry_run: false,
    });
  });

  it('moveSelectionToCluster: POST crops/move with cluster_id', async () => {
    const f = stub({ updated: 2, updated_ids: ['a', 'b'], conflicts: [] });
    const res = await moveSelectionToCluster(SEL, 9, false);
    const [url, init] = f.mock.calls[0]!;
    expect(url).toBe(`${API_PREFIX}/crops/move`);
    expect(JSON.parse(String(init?.body))).toEqual({
      selection: SEL,
      cluster_id: 9,
      dry_run: false,
    });
    expect('updated_ids' in res && res.updated_ids).toEqual(['a', 'b']);
  });

  it('the crop-id exclude/unexclude callers keep their bodies and now read updated_ids', async () => {
    const f = stub({ excluded: 1, updated_ids: ['a'], errors: 0 });
    const ex = await excludeCrops(['a'], 'ignore');
    expect(JSON.parse(String(f.mock.calls[0][1]?.body))).toEqual({
      crop_ids: ['a'],
      reason: 'ignore',
    });
    expect(ex.updated_ids).toEqual(['a']);
    const g = stub({ unexcluded: 1, updated_ids: ['a'], errors: 0 });
    const un = await unexcludeCrops(['a']);
    expect(JSON.parse(String(g.mock.calls[0][1]?.body))).toEqual({ crop_ids: ['a'] });
    expect(un.updated_ids).toEqual(['a']);
  });
});

describe('class identity is by name on the list routes', () => {
  it('getClusters sends class_name repeats and never class_id', async () => {
    const f = stub({ items: [], total: 0 });
    await getClusters({
      class_name: ['widget', 'gadget'],
      exclude_class_name: ['junk'],
      origin: ['sam3'],
      conf_min: 0.5,
    });
    const url = new URL(String(f.mock.calls[0][0]), 'http://x');
    expect(url.searchParams.getAll('class_name')).toEqual(['widget', 'gadget']);
    expect(url.searchParams.getAll('exclude_class_name')).toEqual(['junk']);
    expect(url.searchParams.getAll('origin')).toEqual(['sam3']);
    expect(url.searchParams.get('conf_min')).toBe('0.5');
    expect(url.searchParams.has('class_id')).toBe(false);
  });

  it('getClusters forwards every shared item-filter key (a new key must be listed there too)', async () => {
    // `satisfies` makes this fail to compile when ItemFilterQuery gains a key
    // that ClusterFilter inherits (open_vocab_set and source_prompt are not clusters params), until the key is added here and to getClusters.
    const every = {
      class_name: ['a'],
      exclude_class_name: ['b'],
      conf_min: 0.1,
      conf_max: 0.9,
      min_area: 0.2,
      max_area: 0.8,
      max_rank: 2,
      origin: ['sam3'],
      embedding_state: ['failed'],
      review_status: ['pending'],
    } satisfies Required<Omit<ItemFilterQuery, 'open_vocab_set' | 'source_prompt'>>;
    const f = stub({ items: [], total: 0 });
    await getClusters(every);
    const url = new URL(String(f.mock.calls[0][0]), 'http://x');
    for (const key of Object.keys(every)) {
      expect(url.searchParams.has(key), key).toBe(true);
    }
  });

  it('getRegions sends the shared filter beside its own keys', async () => {
    const f = stub({ items: [], total: 0, page: 1, page_size: 50 });
    await getRegions('/regions', {
      class_name: ['widget'],
      min_area: 0.1,
      status: 'detected',
    });
    const url = new URL(String(f.mock.calls[0][0]), 'http://x');
    expect(url.searchParams.getAll('class_name')).toEqual(['widget']);
    expect(url.searchParams.get('min_area')).toBe('0.1');
    expect(url.searchParams.get('status')).toBe('detected');
  });

  it('getMatchingItemCount sends the filter to stats/dataset and reads the served total_crops', async () => {
    const f = stub({ total_crops: 4 });
    const n = await getMatchingItemCount({
      class_name: ['widget'],
      review_status: ['pending'],
    });
    const url = new URL(String(f.mock.calls[0][0]), 'http://x');
    expect(url.pathname.endsWith('/stats/dataset')).toBe(true);
    expect(url.searchParams.getAll('class_name')).toEqual(['widget']);
    expect(url.searchParams.getAll('review_status')).toEqual(['pending']);
    expect(n).toBe(4);
  });

  it('getMatchingItemCount is null when the response carries no total_crops', async () => {
    stub({});
    expect(await getMatchingItemCount({ class_name: ['widget'] })).toBeNull();
  });
});

describe('exportYolo item_filter', () => {
  it('sends item_filter only when a filter is given and non-empty', async () => {
    const f = stub({});
    await exportYolo({ item_filter: { class_names: ['widget'] } });
    expect(JSON.parse(String(f.mock.calls[0][1]?.body))).toEqual({
      item_filter: { class_names: ['widget'] },
    });
    await exportYolo({ item_filter: {} });
    expect(JSON.parse(String(f.mock.calls[1][1]?.body))).toEqual({});
    await exportYolo({});
    expect(JSON.parse(String(f.mock.calls[2][1]?.body))).toEqual({});
  });
});
