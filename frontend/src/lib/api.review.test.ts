/**
 * Tests for the review-queue wrappers: the queue page, locate, the
 * new-class proposals summary and text search.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { searchCrops } from '$lib/api_cropRead';
import {
  getNewClassProposalsSummary,
  getReviewQueue,
  locateInReviewQueue,
} from '$lib/api_review';

/**
 * Phase 3 (curation-strategy plan §5): getReviewQueue's `filter` argument
 * already accepts arbitrary keys, so StrategyBar's `sort` /
 * `min_mistakenness` / `hide_near_duplicates` need no new plumbing in
 * getReviewQueue itself — just qs()'s existing null-dropping behavior and
 * a pass-through of the new `sort_fallback_reason` + mistakenness fields
 * on the response.
 */
describe('getReviewQueue', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('forwards sort/min_mistakenness/hide_near_duplicates from an arbitrary filter object', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ total: 0, page: 1, page_size: 30, items: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await getReviewQueue('all', 1, 30, {
      sort: 'mistakenness',
      min_mistakenness: 0.5,
      hide_near_duplicates: true,
    });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('sort=mistakenness');
    expect(url).toContain('min_mistakenness=0.5');
    expect(url).toContain('hide_near_duplicates=true');
  });

  it('omits strategy params entirely when the filter object is empty (qs() drops nothing extra)', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ total: 0, page: 1, page_size: 30, items: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await getReviewQueue('all', 1, 30, {});

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).not.toContain('sort=');
    expect(url).not.toContain('min_mistakenness');
    expect(url).not.toContain('hide_near_duplicates');
  });

  it('surfaces sort_fallback_reason from the raw response', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total: 0,
        page: 1,
        page_size: 30,
        items: [],
        sort_fallback_reason: 'mistakenness not backfilled for this pool',
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getReviewQueue('all', 1, 30, { sort: 'mistakenness' });
    expect(res.sort_fallback_reason).toBe('mistakenness not backfilled for this pool');
  });

  it('defaults sort_fallback_reason to null when the server omits it', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ total: 0, page: 1, page_size: 30, items: [] }));
    vi.stubGlobal('fetch', fetchMock);

    const res = await getReviewQueue('all', 1, 30, {});
    expect(res.sort_fallback_reason).toBeNull();
  });

  it('maps mistakenness_score/method/version through onto each item', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total: 1,
        page: 1,
        page_size: 30,
        items: [
          {
            crop_id: 'c1',
            image_path: '/x/y.jpg',
            bbox_norm: [0, 0, 1, 1],
            mistakenness_score: 0.87,
            mistakenness_method: 'mistakenness',
            mistakenness_version: 'v1',
          },
        ],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getReviewQueue('all', 1, 30, {});
    expect(res.items[0]?.mistakenness_score).toBe(0.87);
    expect(res.items[0]?.mistakenness_method).toBe('mistakenness');
    expect(res.items[0]?.mistakenness_version).toBe('v1');
  });

  // dq-queues cutover (2026-09-24): /review/mismatches's `reason` is
  // per-item, not the tab's static description repeated on every row —
  // getReviewQueue must preserve each item's own `reason` verbatim.
  it("maps each item's own reason verbatim, not a single tab-wide value", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total: 2,
        page: 1,
        page_size: 30,
        items: [
          {
            crop_id: 'c1',
            image_path: '/x/y.jpg',
            bbox_norm: [0, 0, 1, 1],
            reason: "VLM said 'widget_e', no registry match",
          },
          {
            crop_id: 'c2',
            image_path: '/x/y2.jpg',
            bbox_norm: [0, 0, 1, 1],
            reason: "VLM said 'wagon-ish', low confidence",
          },
        ],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getReviewQueue('mismatches', 1, 30, {});
    expect(res.items[0]?.reason).toBe("VLM said 'widget_e', no registry match");
    expect(res.items[1]?.reason).toBe("VLM said 'wagon-ish', low confidence");
    expect(res.items[0]?.reason).not.toBe(res.items[1]?.reason);
  });

  // dq-queues cutover: class_confidence/class_confidence_source/
  // vlm_raw_class/vlm_class_empty_reason flow through the review queue
  // mapping same as any other crop-shaped item field.
  it('maps class_confidence/class_confidence_source/vlm_raw_class/vlm_class_empty_reason through', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total: 1,
        page: 1,
        page_size: 30,
        items: [
          {
            crop_id: 'c1',
            image_path: '/x/y.jpg',
            bbox_norm: [0, 0, 1, 1],
            class_confidence: 0.7,
            class_confidence_source: 'vlm',
            vlm_raw_class: 'widget_e',
            vlm_class_empty_reason: null,
          },
        ],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getReviewQueue('mismatches', 1, 30, {});
    expect(res.items[0]?.class_confidence).toBe(0.7);
    expect(res.items[0]?.class_confidence_source).toBe('vlm');
    expect(res.items[0]?.vlm_raw_class).toBe('widget_e');
  });

  it('leaves mistakenness fields null when the server omits them (un-backfilled pool)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total: 1,
        page: 1,
        page_size: 30,
        items: [{ crop_id: 'c1', image_path: '/x/y.jpg', bbox_norm: [0, 0, 1, 1] }],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getReviewQueue('all', 1, 30, {});
    expect(res.items[0]?.mistakenness_score).toBeNull();
  });

  // proposed_class_id/name are served on every
  // crop-shaped item now (item 11) and flow through mapRawCrop — no more
  // review-only special casing — while probe_pred_class_id/needs_new_class/
  // needs_new_class_note (item 14) are still review-item-only extras.
  it('maps proposed_class_id/_name, probe_pred_class_id and needs_new_class through onto each item', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total: 1,
        page: 1,
        page_size: 30,
        items: [
          {
            crop_id: 'c1',
            image_path: '/x/y.jpg',
            bbox_norm: [0, 0, 1, 1],
            proposed_class_id: 12,
            proposed_class_name: 'widget_b',
            probe_pred_class: 'widget_a',
            probe_pred_class_id: 7,
            probe_pred_entropy: 0.42,
            needs_new_class: true,
            needs_new_class_note: 'looks like a boat trailer',
          },
        ],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getReviewQueue('all', 1, 30, {});
    const item = res.items[0];
    expect(item?.proposed_class_id).toBe(12);
    expect(item?.proposed_class_name).toBe('widget_b');
    expect(item?.probe_pred_class).toBe('widget_a');
    expect(item?.probe_pred_class_id).toBe(7);
    expect(item?.probe_pred_entropy).toBe(0.42);
    expect(item?.needs_new_class).toBe(true);
    expect(item?.needs_new_class_note).toBe('looks like a boat trailer');
  });

  it('defaults proposed_class_id/probe_pred_class_id/needs_new_class when the server omits them', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total: 1,
        page: 1,
        page_size: 30,
        items: [{ crop_id: 'c1', image_path: '/x/y.jpg', bbox_norm: [0, 0, 1, 1] }],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getReviewQueue('all', 1, 30, {});
    const item = res.items[0];
    expect(item?.proposed_class_id).toBeNull();
    expect(item?.probe_pred_class_id).toBeNull();
    expect(item?.needs_new_class).toBe(false);
    expect(item?.needs_new_class_note).toBeNull();
  });

  it('surfaces sort_applied from the raw response', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total: 0,
        page: 1,
        page_size: 30,
        items: [],
        sort_applied: 'atypicality_default',
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getReviewQueue('all', 1, 30, {});
    expect(res.sort_applied).toBe('atypicality_default');
  });

  it('defaults sort_applied to null when the server omits it', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ total: 0, page: 1, page_size: 30, items: [] }));
    vi.stubGlobal('fetch', fetchMock);

    const res = await getReviewQueue('all', 1, 30, {});
    expect(res.sort_applied).toBeNull();
  });
});

/**
 * `GET {API_PREFIX}/review/{tab}/locate` — powers the `/review?crop_id=` deep link without paging through
 * the queue by hand.
 */
describe('locateInReviewQueue', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('sends crop_id/page_size/filters and parses an in-queue result', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        crop_id: 'c1',
        in_queue: true,
        rank: 41,
        page: 2,
        page_size: 30,
        total: 109,
        reason: null,
        sort_applied: 'atypicality',
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await locateInReviewQueue('all', 'c1', 30, { class_id: 5 });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('/review/all/locate');
    expect(url).toContain('crop_id=c1');
    expect(url).toContain('page_size=30');
    expect(url).toContain('class_id=5');
    expect(res).toEqual({
      crop_id: 'c1',
      in_queue: true,
      rank: 41,
      page: 2,
      page_size: 30,
      total: 109,
      reason: null,
      sort_applied: 'atypicality',
      sort_fallback_reason: null,
    });
  });

  it('parses a not-in-queue result with a reason and null page/rank', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        crop_id: 'c1',
        in_queue: false,
        rank: null,
        page: null,
        page_size: 30,
        total: 0,
        reason: 'filtered_out',
        sort_applied: 'uncertainty_entropy',
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await locateInReviewQueue('uncertainty', 'c1', 30);
    expect(res.in_queue).toBe(false);
    expect(res.page).toBeNull();
    expect(res.rank).toBeNull();
    expect(res.reason).toBe('filtered_out');
  });
});

/**
 * `GET {API_PREFIX}/review/new_class_proposals/summary` — the aggregate `/classes`'s Proposals section renders.
 */
describe('getNewClassProposalsSummary', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('parses total_pending and top_terms', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total_pending: 8,
        top_terms: [
          {
            label: 'boat',
            count: 5,
            sample_crop_ids: ['a', 'b'],
            flag: null,
            class_id: null,
          },
        ],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getNewClassProposalsSummary();
    expect(res.total_pending).toBe(8);
    expect(res.top_terms).toEqual([
      {
        label: 'boat',
        count: 5,
        sample_crop_ids: ['a', 'b'],
        flag: null,
        class_id: null,
      },
    ]);
  });

  it('defaults to zero/empty when the backend is degraded (e.g. an opensearch outage)', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ detail: 'opensearch unavailable' }));
    vi.stubGlobal('fetch', fetchMock);

    const res = await getNewClassProposalsSummary();
    expect(res.total_pending).toBe(0);
    expect(res.top_terms).toEqual([]);
    expect(res.flagged_terms).toEqual([]);
    expect(res.without_term).toBe(0);
    expect(res.term_rules).toBeNull();
  });

  // DQ-M11 fix (dq-queues cutover, 2026-09-24): flagged_terms/
  // without_term/term_rules are new response fields.
  it('parses flagged_terms with their flag/class_id, without_term and term_rules', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total_pending: 191,
        without_term: 12,
        top_terms: [],
        flagged_terms: [
          {
            label: 'widget_c',
            count: 89,
            sample_crop_ids: ['a'],
            flag: 'generic_parent',
            class_id: null,
          },
          {
            label: 'widget_b',
            count: 4,
            sample_crop_ids: ['b'],
            flag: 'existing_class',
            class_id: 67,
          },
          {
            label: 'abstract_blur',
            count: 2,
            sample_crop_ids: [],
            flag: 'non_object',
            class_id: null,
          },
        ],
        term_rules: {
          generic_terms: ['container', 'widget_c'],
          non_object_terms: ['abstract_blur'],
          registry_groups_are_generic: true,
          existing_classes_flagged: true,
          generic_terms_env: 'OP_NEW_CLASS_GENERIC_TERMS',
          non_object_terms_env: 'OP_NEW_CLASS_NON_OBJECT_TERMS',
        },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getNewClassProposalsSummary();
    expect(res.without_term).toBe(12);
    expect(res.flagged_terms).toHaveLength(3);
    expect(res.flagged_terms[0]).toEqual({
      label: 'widget_c',
      count: 89,
      sample_crop_ids: ['a'],
      flag: 'generic_parent',
      class_id: null,
    });
    expect(res.flagged_terms[1].flag).toBe('existing_class');
    expect(res.flagged_terms[1].class_id).toBe(67);
    expect(res.term_rules?.generic_terms).toEqual(['container', 'widget_c']);
  });
});

/**
 * searchCrops (P2-14 semantic text search) mirrors getReviewQueue's
 * response-shape handling — same qs()-forwarding, same mapRawCrop
 * normalization — but points at `GET {API_PREFIX}/search/text` and adds the
 * per-item similarity score instead of the review-queue's reason/
 * proposed-class fields.
 */
describe('searchCrops', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('sends q/page/page_size and hits GET {API_PREFIX}/search/text', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ total: 0, page: 1, page_size: 30, items: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await searchCrops('red widget_a', 1, 30);

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain(`${API_PREFIX}/search/text`);
    expect(url).toContain('q=red+widget_a');
    expect(url).toContain('page=1');
    expect(url).toContain('page_size=30');
  });

  it('forwards extra filter params (e.g. cluster_id, review tab filters) unchanged', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ total: 0, page: 1, page_size: 30, items: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await searchCrops('blue truck', 1, 30, { cluster_id: 42, max_rank: 1 });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('cluster_id=42');
    expect(url).toContain('max_rank=1');
  });

  it('maps the live backend field `semantic_score` through onto each item (OpenProcessor _hydrate_item)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total: 1,
        page: 1,
        page_size: 30,
        items: [
          {
            crop_id: 'c1',
            image_path: '/x/y.jpg',
            bbox_norm: [0, 0, 1, 1],
            semantic_score: 0.73,
          },
        ],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await searchCrops('red widget_a', 1, 30);
    expect(res.items[0]?.similarity_score).toBe(0.73);
  });

  it('defaults similarity_score to 0 when the server omits both fields', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total: 1,
        page: 1,
        page_size: 30,
        items: [{ crop_id: 'c1', image_path: '/x/y.jpg', bbox_norm: [0, 0, 1, 1] }],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await searchCrops('red widget_a', 1, 30);
    expect(res.items[0]?.similarity_score).toBe(0);
  });

  it('returns an empty result set on a zero-hit response (empty-state input)', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ total: 0, page: 1, page_size: 30, items: [] }));
    vi.stubGlobal('fetch', fetchMock);

    const res = await searchCrops('zzzznonexistentqueryzzzz', 1, 30);
    expect(res.items).toEqual([]);
    expect(res.total).toBe(0);
  });
});
