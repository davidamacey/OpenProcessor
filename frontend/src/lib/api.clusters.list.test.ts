/**
 * `getClusters` / `getClusterFacets`: the served sort and card filters go out
 * as query parameters unchanged, and the new card and response fields come
 * back as served (no client rank, no `label_purity` fallback).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { getClusterFacets, getClusters } from '$lib/api_clusters';

const UID = '0b9f9a38-0000-4000-8000-000000000007';

const json = (body: unknown) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });

function card(extra: Record<string, unknown> = {}) {
  return {
    cluster_uid: '0b9f9a38-0000-4000-8000-000000000001',
    size: 40,
    validated_count: 4,
    labelled_count: 20,
    dominant_class_name: null,
    dominant_count: 0,
    label_agreement: 0.75,
    labelled_share: 0.5,
    is_unlabeled: true,
    updated_at: null,
    representatives: [],
    detector_top_class: 'widget_x',
    detector_top_share: 0.62,
    detector_conf_mean: 0.81,
    validation_state: 'partial',
    review_counts: { all: 7, outliers: 3 },
    needs_review_count: 7,
    flags: ['mixed_detector_classes'],
    ...extra,
  };
}

const LIST = {
  items: [card()],
  total: 1,
  representatives_offset: 0,
  representatives_limit: 24,
  sort: 'needs_review_desc',
  sort_options: [{ id: 'needs_review_desc', label: 'Needs review first' }],
  flag_vocabulary: [{ id: 'mixed_detector_classes', label: 'Mixed', description: 'why' }],
  truncated: true,
  empty_reason: null,
  empty_message: null,
  pending_partition: null,
  hidden: { test_holdout: 25, excluded: 4 },
};

function stubFetch(body: unknown = LIST) {
  const fetchMock = vi.fn().mockResolvedValue(json(body));
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}
const urlOf = (m: ReturnType<typeof vi.fn>, i = 0) =>
  new URL(String(m.mock.calls[i]![0]), 'http://x');

afterEach(() => vi.unstubAllGlobals());

describe('getClusters query', () => {
  it('sends the sort and every card filter as served parameters', async () => {
    const m = stubFetch();
    await getClusters({
      sort: 'label_agreement_asc',
      dominant_class: ['class_a', 'class_b'],
      detector_class: ['widget_x'],
      min_size: 5,
      max_size: 90,
      validation_state: 'none',
      flag: ['unvalidated', 'low_label_agreement'],
      needs_review: true,
    });
    const q = urlOf(m).searchParams;
    expect(q.get('sort')).toBe('label_agreement_asc');
    expect(q.getAll('dominant_class')).toEqual(['class_a', 'class_b']);
    expect(q.getAll('detector_class')).toEqual(['widget_x']);
    expect(q.get('min_size')).toBe('5');
    expect(q.get('max_size')).toBe('90');
    expect(q.get('validation_state')).toBe('none');
    expect(q.getAll('flag')).toEqual(['unvalidated', 'low_label_agreement']);
    expect(q.get('needs_review')).toBe('true');
  });

  it('sends the review queue so the count matches the other views', async () => {
    const m = stubFetch();
    await getClusters({ review_tab: 'check_machine', class_name: ['person'] });
    const q = urlOf(m).searchParams;
    expect(q.get('review_tab')).toBe('check_machine');
    expect(q.get('class_name')).toBe('person');
  });

  it('sends no sort and no filter when none is chosen', async () => {
    const m = stubFetch();
    await getClusters();
    const q = urlOf(m).searchParams;
    for (const k of ['sort', 'dominant_class', 'min_size', 'flag', 'needs_review']) {
      expect(q.has(k), k).toBe(false);
    }
  });

  it('pages representatives with the offset and limit it was given', async () => {
    const m = stubFetch();
    await getClusters({ representatives_offset: 24, representatives_limit: 24 });
    const q = urlOf(m).searchParams;
    expect(q.get('offset')).toBe('24');
    expect(q.get('limit')).toBe('24');
  });
});

describe('getClusters response', () => {
  it('maps the new card fields verbatim', async () => {
    stubFetch();
    const res = await getClusters();
    expect(res.items[0]).toMatchObject({
      detector_top_class: 'widget_x',
      detector_top_share: 0.62,
      detector_conf_mean: 0.81,
      validation_state: 'partial',
      review_counts: { all: 7, outliers: 3 },
      needs_review_count: 7,
      flags: ['mixed_detector_classes'],
      label_agreement: 0.75,
    });
  });

  it('carries the sort, sort options, flag vocabulary and truncated flag', async () => {
    stubFetch();
    const res = await getClusters();
    expect(res.sort).toBe('needs_review_desc');
    expect(res.sort_options).toEqual(LIST.sort_options);
    expect(res.flag_vocabulary).toEqual(LIST.flag_vocabulary);
    expect(res.truncated).toBe(true);
    expect(res.total).toBe(1);
  });

  it('does not read label_purity (removed from the wire)', async () => {
    stubFetch({ ...LIST, items: [card({ label_agreement: null, label_purity: 0.9 })] });
    const res = await getClusters();
    expect(res.items[0]).toMatchObject({ label_agreement: null, dominant_pct: null });
    expect(res.items[0]).not.toHaveProperty('label_purity');
  });
});

describe('getClusterFacets', () => {
  it('reads the cluster facets route and returns the body as served', async () => {
    const facets = {
      cluster_uid: UID,
      total: 3,
      detector_classes: [{ value: 'widget_x', count: 2, label: null }],
      class_sources: [],
      validation: { validated: 1, unvalidated: 2 },
      review_tabs: [{ value: 'outliers', count: 1, label: 'Outliers' }],
      detector_confidence: [{ min: 0.9, max: null, count: 3 }],
      detector_confidence_missing: 0,
    };
    const m = stubFetch(facets);
    const res = await getClusterFacets(UID);
    expect(String(m.mock.calls[0]![0])).toBe(`${API_PREFIX}/clusters/${UID}/facets`);
    expect(res).toEqual(facets);
  });
});

describe('getClusters empty state', () => {
  it('passes the served reason, text and pending partition through', async () => {
    stubFetch({
      ...LIST,
      items: [],
      total: 0,
      empty_reason: 'pending_activation',
      empty_message: 'A clustering run finished but is not serving yet.',
      pending_partition: { partition_id: 'p-1', clusters: 9, items: 120 },
    });
    const res = await getClusters();
    expect(res.empty_reason).toBe('pending_activation');
    expect(res.empty_message).toBe('A clustering run finished but is not serving yet.');
    expect(res.pending_partition).toEqual({
      partition_id: 'p-1',
      clusters: 9,
      items: 120,
    });
  });
});

describe('getClusters hidden counts', () => {
  it('returns the served counts of what the filter leaves out', async () => {
    stubFetch();
    const res = await getClusters();
    expect(res.hidden).toEqual({ test_holdout: 25, excluded: 4 });
  });
});
