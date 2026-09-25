/**
 * Tests for ApiError's message composition.
 *
 * Every UI callsite renders `(e as Error).message`, so the server's reason
 * has to be baked into the message or the operator never sees it.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  apiBase,
  API_PREFIX,
  ApiError,
  cancelScores,
  cancelSelect,
  computeScores,
  getAutoLabelJobStatus,
  getCluster,
  getClusters,
  getClassRegistryUrl,
  getCurationSettings,
  getDataYamlUrl,
  getManifestUrl,
  getMethods,
  getNewClassProposalsSummary,
  getReviewQueue,
  getScoresCoverage,
  getScoresStatus,
  getSelectStatus,
  getVizProjection,
  locateInReviewQueue,
  normalizeApiPrefix,
  pollAutoLabelJob,
  putCurationDefaults,
  rebuildVizProjection,
  runVlmOnCluster,
  searchCrops,
  selectDiverse,
  startAutoLabel,
} from './api';
import { FALLBACK_METHODS } from './strategies';

const URL = `http://localhost:4603${API_PREFIX}/crops/batch_label`;

describe('ApiError', () => {
  it("appends a FastAPI 'detail' string to the message", () => {
    const e = new ApiError(422, URL, { detail: 'region bbox outside crop envelope' });
    expect(e.detail).toBe('region bbox outside crop envelope');
    expect(e.message).toBe(`API 422 ${URL} — region bbox outside crop envelope`);
  });

  it("falls back to a 'message' property", () => {
    const e = new ApiError(400, URL, { message: 'hotkey already bound' });
    expect(e.message).toContain('hotkey already bound');
  });

  it('accepts a plain-text body', () => {
    const e = new ApiError(502, URL, 'upstream unavailable');
    expect(e.detail).toBe('upstream unavailable');
  });

  it('truncates a long detail to 200 chars', () => {
    const e = new ApiError(422, URL, { detail: 'x'.repeat(500) });
    expect(e.detail).toHaveLength(200);
    expect(e.detail?.endsWith('…')).toBe(true);
  });

  it('leaves the message unadorned when there is no usable detail', () => {
    expect(new ApiError(500, URL, null).message).toBe(`API 500 ${URL}`);
    expect(new ApiError(500, URL, { detail: '   ' }).message).toBe(`API 500 ${URL}`);
    expect(new ApiError(500, URL, { detail: [{ loc: ['body'] }] }).message).toBe(
      `API 500 ${URL}`,
    );
  });

  it('honors an explicit message override', () => {
    const e = new ApiError(404, URL, { detail: 'nope' }, 'custom');
    expect(e.message).toBe('custom');
    expect(e.detail).toBe('nope');
  });
});

/**
 * getMethods() must never throw — {API_PREFIX}/methods is optional capability
 * discovery (plan §5.3). A 404 or any other failure resolves to the
 * hardcoded FALLBACK_METHODS instead of rejecting, so a backend that
 * hasn't shipped the endpoint yet can't break app boot.
 */
describe('getMethods', () => {
  const jsonResponse = (body: unknown, init: ResponseInit = {}) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
      ...init,
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns the real parsed response on success', async () => {
    // Real {API_PREFIX}/methods wire shape (confirmed 2026-09-10 against
    // strategy_registry.py's get_registry()): a flat `strategies` array,
    // each entry carrying an `axis` field — not four separate top-level
    // arrays.
    const serverBody = {
      strategies: [
        {
          id: 'ivf',
          axis: 'cluster',
          label: 'FAISS IVF-512 (production)',
          status: 'stable',
          default: true,
        },
        {
          id: 'default',
          axis: 'sort',
          label: 'Recent first',
          status: 'stable',
          default: true,
        },
        {
          id: 'uncertainty',
          axis: 'sort',
          label: 'Uncertainty margin',
          status: 'experimental',
        },
      ],
      flags: {},
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(serverBody));
    vi.stubGlobal('fetch', fetchMock);

    const result = await getMethods();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(result.cluster_methods).toEqual([
      { id: 'ivf', label: 'FAISS IVF-512 (production)', status: 'stable', default: true },
    ]);
    expect(result.review_sorts).toEqual([
      { id: 'default', label: 'Recent first', status: 'stable', default: true },
      {
        id: 'uncertainty',
        label: 'Uncertainty margin',
        status: 'experimental',
        default: undefined,
      },
    ]);
    // Real backend response, not the hardcoded fallback.
    expect(result).not.toEqual(FALLBACK_METHODS);
  });

  it('resolves to FALLBACK_METHODS on a 404, without throwing or retrying', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'not found' }), { status: 404 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await expect(getMethods()).resolves.toEqual(FALLBACK_METHODS);
    // No retry on 4xx — matches apiFetch's documented "don't retry on 4xx" rule.
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('resolves to FALLBACK_METHODS on a network failure, after the normal 5xx/network retry budget', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError('fetch failed'));
    vi.stubGlobal('fetch', fetchMock);

    await expect(getMethods()).resolves.toEqual(FALLBACK_METHODS);
    // apiFetch's retry loop: 1 initial + 3 retries = 4 attempts.
    expect(fetchMock).toHaveBeenCalledTimes(4);
  }, 10_000);

  it('resolves to FALLBACK_METHODS (not throws) on a malformed 200 body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('not json', {
        status: 200,
        headers: { 'content-type': 'text/plain' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    // A plain-text 200 body parses to a string, which parseMethodsResponse
    // treats as "not a usable object" and degrades to empty lists — not a
    // crash, and distinct from the true-failure fallback path.
    const result = await getMethods();
    expect(result).toEqual({
      cluster_methods: [],
      review_sorts: [],
      overlays: [],
      scores: [],
      dataset_exports: [],
      detection_profiles: [],
      prompt_packs: [],
    });
  });

  it('propagates a caller-initiated abort instead of swallowing it into the fallback', async () => {
    const ctrl = new AbortController();
    const fetchMock = vi.fn().mockImplementation((_url: string, init: RequestInit) => {
      return new Promise((_resolve, reject) => {
        init.signal?.addEventListener('abort', () => {
          reject(new DOMException('Aborted', 'AbortError'));
        });
      });
    });
    vi.stubGlobal('fetch', fetchMock);

    const p = getMethods(ctrl.signal);
    ctrl.abort();
    await expect(p).rejects.toMatchObject({ name: 'AbortError' });
  });
});

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

  // 2026-09-24 logic-moves W5: proposed_class_id/name are served on every
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
 * `GET {API_PREFIX}/review/{tab}/locate` (item 10, 2026-09-24 logic-moves
 * W5) — powers the `/review?crop_id=` deep link without paging through
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
 * `GET {API_PREFIX}/review/new_class_proposals/summary` (2026-09-24
 * logic-moves W5) — the aggregate `/classes`'s Proposals section renders.
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

  it('maps similarity_score through onto each item', async () => {
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
            similarity_score: 0.91,
          },
        ],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await searchCrops('red widget_a', 1, 30);
    expect(res.items[0]?.similarity_score).toBe(0.91);
    expect(res.items[0]?.id).toBe('c1');
  });

  it('maps the live backend field `semantic_score` through onto each item (openprocessor _hydrate_item)', async () => {
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

  it('prefers similarity_score over semantic_score over score when more than one is present', async () => {
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
            similarity_score: 0.91,
            semantic_score: 0.73,
            score: 0.5,
          },
        ],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const res1 = await searchCrops('red widget_a', 1, 30);
    expect(res1.items[0]?.similarity_score).toBe(0.91);

    const fetchMock2 = vi.fn().mockResolvedValue(
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
            score: 0.5,
          },
        ],
      }),
    );
    vi.stubGlobal('fetch', fetchMock2);
    const res2 = await searchCrops('red widget_a', 1, 30);
    expect(res2.items[0]?.similarity_score).toBe(0.73);
  });

  it('falls back to a bare score field if the server sends that instead', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        total: 1,
        page: 1,
        page_size: 30,
        items: [
          { crop_id: 'c1', image_path: '/x/y.jpg', bbox_norm: [0, 0, 1, 1], score: 0.5 },
        ],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await searchCrops('red widget_a', 1, 30);
    expect(res.items[0]?.similarity_score).toBe(0.5);
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

/**
 * getCluster's `order` param is forwarded verbatim to `{API_PREFIX}/crops?order=`
 * (broadened from a fixed `'outliers'` literal so a future
 * `{API_PREFIX}/methods`-reported order id doesn't require touching this
 * signature — see the comment on `order` in api.ts). An id the backend
 * doesn't recognize should be harmless: qs() still sends it, and callers
 * are responsible for only offering ids `{API_PREFIX}/methods` actually reports.
 */
describe('getCluster order param', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('forwards an arbitrary order id to {API_PREFIX}/crops without special-casing it client-side', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.startsWith(`${API_PREFIX}/crops`)) {
        return Promise.resolve(
          jsonResponse({ total: 0, page: 1, page_size: 60, crops: [] }),
        );
      }
      return Promise.resolve(jsonResponse({ items: [] }));
    });
    vi.stubGlobal('fetch', fetchMock);

    await getCluster(42, 1, 60, undefined, { order: 'mistakenness' });

    const cropsUrl = fetchMock.mock.calls
      .map((c) => c[0] as string)
      .find((u) => u.startsWith(`${API_PREFIX}/crops`));
    expect(cropsUrl).toContain('order=mistakenness');
  });

  it('omits order entirely when null (unchanged default behavior)', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.startsWith(`${API_PREFIX}/crops`)) {
        return Promise.resolve(
          jsonResponse({ total: 0, page: 1, page_size: 60, crops: [] }),
        );
      }
      return Promise.resolve(jsonResponse({ items: [] }));
    });
    vi.stubGlobal('fetch', fetchMock);

    await getCluster(42, 1, 60, undefined, { order: null });

    const cropsUrl = fetchMock.mock.calls
      .map((c) => c[0] as string)
      .find((u) => u.startsWith(`${API_PREFIX}/crops`));
    expect(cropsUrl).not.toContain('order');
  });
});

/**
 * `k` (curation-strategy plan Phase 4 — "how many diverse crops?", forwarded
 * alongside `order=diverse`). Same forward-verbatim contract as `order`:
 * getCluster doesn't validate the id/count pair, it just plumbs whatever the
 * caller (gated by {API_PREFIX}/methods, see strategies.test.ts's
 * isDiverseOverlayAvailable coverage) decided to send.
 */
describe('getCluster k param', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function stubCrops(body: unknown) {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.startsWith(`${API_PREFIX}/crops`))
        return Promise.resolve(jsonResponse(body));
      return Promise.resolve(jsonResponse({ items: [] }));
    });
    vi.stubGlobal('fetch', fetchMock);
    return fetchMock;
  }

  it('forwards k to {API_PREFIX}/crops when set alongside order=diverse', async () => {
    const fetchMock = stubCrops({ total: 0, page: 1, page_size: 60, crops: [] });

    await getCluster(42, 1, 60, undefined, { order: 'diverse', k: 120 });

    const cropsUrl = fetchMock.mock.calls
      .map((c) => c[0] as string)
      .find((u) => u.startsWith(`${API_PREFIX}/crops`));
    expect(cropsUrl).toContain('order=diverse');
    expect(cropsUrl).toContain('k=120');
  });

  it('omits k entirely when null/undefined (qs() drops it, no ?k= at all)', async () => {
    const fetchMock = stubCrops({ total: 0, page: 1, page_size: 60, crops: [] });

    await getCluster(42, 1, 60, undefined, { order: null, k: null });

    const cropsUrl = fetchMock.mock.calls
      .map((c) => c[0] as string)
      .find((u) => u.startsWith(`${API_PREFIX}/crops`));
    expect(cropsUrl).not.toContain('k=');

    fetchMock.mockClear();
    await getCluster(42, 1, 60, undefined, {});
    const cropsUrl2 = fetchMock.mock.calls
      .map((c) => c[0] as string)
      .find((u) => u.startsWith(`${API_PREFIX}/crops`));
    expect(cropsUrl2).not.toContain('k=');
  });

  it('surfaces order_method/order_version/n_pool when the server sends them', async () => {
    stubCrops({
      total: 500,
      page: 1,
      page_size: 60,
      crops: [],
      method: 'kcenter_greedy',
      version: '1',
      n_pool: 4832,
    });

    const res = await getCluster(42, 1, 60, undefined, { order: 'diverse', k: 60 });

    expect(res.crops.order_method).toBe('kcenter_greedy');
    expect(res.crops.order_version).toBe('1');
    expect(res.crops.n_pool).toBe(4832);
  });

  it('defaults order_method/order_version/n_pool to null when the server omits them', async () => {
    stubCrops({ total: 0, page: 1, page_size: 60, crops: [] });

    const res = await getCluster(42, 1, 60, undefined, { order: null });

    expect(res.crops.order_method).toBeNull();
    expect(res.crops.order_version).toBeNull();
    expect(res.crops.n_pool).toBeNull();
  });
});

/**
 * getVizProjection() — curation-strategy plan Phase 5
 * (docs/curation-strategy-plan-2026-09.md §2.7/§5.6). Never rejects
 * (same contract as getMethods): `{API_PREFIX}/viz/projection` may not exist yet
 * (the openprocessor Phase 5 branch lands independently) and the UMAP
 * purity gate may mean the capability never ships at all — a fetch
 * failure here must degrade `EmbeddingPlot` to its pending/empty state,
 * never crash the page it replaced the grid on.
 *
 * The real wire shape (confirmed 2026-09-10 against
 * `embedding_viz.get_cached_projection`) is `{status: 'not_built'}` when
 * nothing has been fit, or `{points, projection_version, fitted_at,
 * stale}` otherwise — there is no `built`/`built_at`/`version` on the
 * wire; those were an earlier, wrong guess. `built` is this file's own
 * derived convenience field.
 */
describe('getVizProjection', () => {
  const jsonResponse = (body: unknown, init: ResponseInit = {}) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
      ...init,
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('parses a well-formed built response, dropping malformed points', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        points: [
          { crop_id: 'a', x: 1.5, y: -2.3, cluster_id: 17, class_name: 'widget_a' },
          { crop_id: 'b', x: 0, y: 0, cluster_id: null, class_name: null },
          // Malformed entries — must be dropped, not crash the whole parse.
          { crop_id: '', x: 1, y: 1 },
          { x: 1, y: 1 },
          { crop_id: 'c', x: 'nope', y: 1 },
          null,
          'garbage',
        ],
        projection_version: '1',
        fitted_at: '2026-09-10T00:00:00Z',
        stale: false,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getVizProjection({ max_points: 100 });

    expect(res.built).toBe(true);
    expect(res.points).toEqual([
      {
        crop_id: 'a',
        x: 1.5,
        y: -2.3,
        cluster_id: 17,
        class_name: 'widget_a',
        class_source: null,
      },
      {
        crop_id: 'b',
        x: 0,
        y: 0,
        cluster_id: null,
        class_name: null,
        class_source: null,
      },
    ]);
    expect(res.total).toBe(2);
    expect(res.projection_version).toBe('1');
    expect(res.fitted_at).toBe('2026-09-10T00:00:00Z');
    expect(res.stale).toBe(false);

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain(`${API_PREFIX}/viz/projection`);
    expect(url).toContain('max_points=100');
  });

  it('forwards cluster_id/class_id and omits unset params', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        points: [],
        projection_version: '1',
        fitted_at: null,
        stale: false,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await getVizProjection({ cluster_id: 42 });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('cluster_id=42');
    expect(url).not.toContain('class_id');
    expect(url).not.toContain('max_points');
  });

  it("reports built:false when the server says status: 'not_built'", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ status: 'not_built' }));
    vi.stubGlobal('fetch', fetchMock);
    expect((await getVizProjection()).built).toBe(false);
  });

  it('reports stale:true when the server flags a partial-coverage projection', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        points: [],
        projection_version: '2',
        fitted_at: '2026-09-01T00:00:00Z',
        stale: true,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    expect((await getVizProjection()).stale).toBe(true);
  });

  it('resolves to the empty/pending fallback on a 404, without throwing or retrying', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'not found' }), { status: 404 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getVizProjection();
    expect(res).toEqual({
      points: [],
      total: 0,
      built: false,
      fitted_at: null,
      projection_version: null,
      stale: false,
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('resolves to the empty/pending fallback on a network failure, after the retry budget', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError('fetch failed'));
    vi.stubGlobal('fetch', fetchMock);

    const res = await getVizProjection();
    expect(res.built).toBe(false);
    expect(res.points).toEqual([]);
    expect(fetchMock).toHaveBeenCalledTimes(4);
  }, 10_000);

  it('resolves to the empty/pending fallback (not throws) on a malformed 200 body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('not json', {
        status: 200,
        headers: { 'content-type': 'text/plain' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getVizProjection();
    expect(res.points).toEqual([]);
    expect(res.built).toBe(false);
  });

  it('propagates a caller-initiated abort instead of swallowing it into the fallback', async () => {
    const ctrl = new AbortController();
    const fetchMock = vi.fn().mockImplementation((_url: string, init: RequestInit) => {
      return new Promise((_resolve, reject) => {
        init.signal?.addEventListener('abort', () => {
          reject(new DOMException('Aborted', 'AbortError'));
        });
      });
    });
    vi.stubGlobal('fetch', fetchMock);

    const p = getVizProjection(undefined, ctrl.signal);
    ctrl.abort();
    await expect(p).rejects.toMatchObject({ name: 'AbortError' });
  });
});

/**
 * rebuildVizProjection() — the real `embedding_viz._JobState` is flat
 * (confirmed 2026-09-10), not the nested `{running, result: {...}}`
 * shape an earlier version of this file guessed: `status` is a string
 * enum, timestamps are unix-epoch numbers (0 when unset), and
 * `n_written`/`projection_version` are top-level fields.
 */
describe('rebuildVizProjection', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('POSTs to {API_PREFIX}/viz/projection/rebuild and returns the job snapshot', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        job_id: 'viz-rebuild-1',
        status: 'running',
        scope: 'all',
        cluster_id: null,
        n_pool: 1200,
        n_written: 0,
        started_at: 1757462400,
        finished_at: 0,
        error: null,
        projection_version: null,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await rebuildVizProjection();

    expect(res.status).toBe('running');
    expect(res.job_id).toBe('viz-rebuild-1');
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain(`${API_PREFIX}/viz/projection/rebuild`);
    expect(init.method).toBe('POST');
  });

  it('propagates a 5xx failure rather than swallowing it (unlike getVizProjection)', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'busy' }), { status: 503 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await expect(rebuildVizProjection()).rejects.toBeInstanceOf(ApiError);
  }, 10_000);
});

/**
 * Regression test for the /export download-button filename mismatch:
 * getDataYamlUrl() used to request `data_v7.yaml`, which the export service
 * never writes (the real on-disk file is `data.yaml`), so the download would
 * 404 against the real `{API_PREFIX}/export/registry/{artifact}` backend contract.
 * Pin all three exact URLs so this can't silently regress.
 */
describe('registry download URL builders', () => {
  it('getClassRegistryUrl() points at the real class_registry.json filename', () => {
    expect(getClassRegistryUrl()).toBe(
      `${apiBase}${API_PREFIX}/export/registry/class_registry.json`,
    );
  });

  it('getDataYamlUrl() points at the real data.yaml filename (not data_v7.yaml)', () => {
    expect(getDataYamlUrl()).toBe(`${apiBase}${API_PREFIX}/export/registry/data.yaml`);
  });

  it('getManifestUrl() points at the real manifest.json filename', () => {
    expect(getManifestUrl()).toBe(
      `${apiBase}${API_PREFIX}/export/registry/manifest.json`,
    );
  });
});

/**
 * P2-10: `/review`'s diverse overlay (`POST {API_PREFIX}/select/diverse`). The
 * endpoint answers 200 (small pool, `crop_ids` ready now), 202 (large
 * pool — job enqueued, poll `getSelectStatus`), 400 (feature disabled),
 * or 409 (singleton job already running elsewhere) — the last two are
 * expected states the UI treats as routine, not toast-worthy failures, so
 * `selectDiverse` must resolve a typed result for them rather than throw.
 */
describe('selectDiverse', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns a "ready" result for a 200 response with crop_ids', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          crop_ids: ['a', 'b'],
          method: 'k_center_greedy',
          version: 'v1',
          n_pool: 2,
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await selectDiverse({ review_tab: 'all' }, 100);
    expect(res).toEqual({
      kind: 'ready',
      selection: {
        crop_ids: ['a', 'b'],
        method: 'k_center_greedy',
        version: 'v1',
        n_pool: 2,
      },
    });
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain(`${API_PREFIX}/select/diverse`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({
      scope: { review_tab: 'all' },
      k: 100,
    });
  });

  it('returns a "job" result for a 202 response with job_id', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ job_id: 'job-123', status: 'running' }), {
        status: 202,
        headers: { 'content-type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await selectDiverse({ review_tab: 'all' }, 500);
    expect(res).toEqual({ kind: 'job', job_id: 'job-123' });
  });

  it('returns "disabled" (not a throw) on a 400', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: 'diverse selection disabled' }), {
        status: 400,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await selectDiverse({ review_tab: 'all' }, 100);
    expect(res).toEqual({ kind: 'disabled' });
  });

  it('returns "already_running" (not a throw) on a 409', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'job already running' }), { status: 409 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    const res = await selectDiverse({ review_tab: 'all' }, 100);
    expect(res).toEqual({ kind: 'already_running' });
  });

  it('still throws on an unrelated 4xx/5xx', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'boom' }), { status: 422 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await expect(selectDiverse({ review_tab: 'all' }, 100)).rejects.toBeInstanceOf(
      ApiError,
    );
  });

  it('forwards seed_crop_id only when given', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ crop_ids: [], method: 'm', version: 'v', n_pool: 0 }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    await selectDiverse({ cluster_id: 5 }, 10, 'seed-crop-1');
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({
      scope: { cluster_id: 5 },
      k: 10,
      seed_crop_id: 'seed-crop-1',
    });
  });
});

describe('getSelectStatus / cancelSelect', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('parses a running-job status payload', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ job_id: 'job-1', status: 'running' }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const st = await getSelectStatus();
    expect(st.status).toBe('running');
    expect(st.job_id).toBe('job-1');
  });

  it('parses a completed-job status payload, result nested (real selection/job.py shape)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          status: 'completed',
          result: {
            crop_ids: ['x', 'y'],
            method: 'kcenter_greedy',
            version: 'v1',
            n_pool: 2,
          },
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const st = await getSelectStatus();
    expect(st.status).toBe('completed');
    expect(st.result).toEqual({
      crop_ids: ['x', 'y'],
      method: 'kcenter_greedy',
      version: 'v1',
      n_pool: 2,
    });
  });

  it('leaves result null for a failed/cancelled job', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          status: 'failed',
          error: 'selection job heartbeat stale (34.6s ago)',
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const st = await getSelectStatus();
    expect(st.status).toBe('failed');
    expect(st.result).toBeNull();
    expect(st.error).toBe('selection job heartbeat stale (34.6s ago)');
  });

  it('cancelSelect POSTs to {API_PREFIX}/select/cancel', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);

    await cancelSelect();
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain(`${API_PREFIX}/select/cancel`);
    expect(init.method).toBe('POST');
  });
});

/**
 * Scoped VLM-assisted labeling (2026-09-20 contract — confirmed live
 * 2026-09-21 against a real OpenProcessor backend via the actual UI,
 * see docs/design/vlm-scoped-labeling-assist-plan-2026-09-20.md
 * §1.3/§5.3). `class_id`/`detection_profile`/`prompt_pack` are optional
 * query params on the existing start call; `qs()` drops null/undefined so
 * an unscoped call must be byte-identical to the pre-scope request.
 */
describe('startAutoLabel', () => {
  const jobResponse = () =>
    new Response(
      JSON.stringify({
        job_id: 'job-1',
        status: 'idle',
        stage: '',
        processed: 0,
        total: 0,
        started_at: 0,
        finished_at: 0,
        error: null,
        result: {},
        args: {},
        eta_seconds: null,
        elapsed_seconds: 0,
      }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    );

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('sends no scoping params when the scope is untouched (byte-identical to the pre-scope request)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await startAutoLabel({ train_clusters: true, vlm_concurrency: 16 });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).not.toContain('class_id');
    expect(url).not.toContain('detection_profile');
    expect(url).not.toContain('prompt_pack');
  });

  it('forwards class_id when the run is scoped to one class', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await startAutoLabel({ class_id: 7 });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('class_id=7');
  });

  it('drops an explicitly null class_id rather than sending class_id=null', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await startAutoLabel({ class_id: null });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).not.toContain('class_id');
    expect(url).not.toContain('null');
  });

  it('forwards prompt_pack when selected', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await startAutoLabel({ prompt_pack: 'warehouse_v1' });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('prompt_pack=warehouse_v1');
  });

  it('composes the path from API_PREFIX', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await startAutoLabel({});

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain(`${API_PREFIX}/pipeline/auto_label/start`);
  });

  // G5: run_vlm was never sent at all before this change, so a scoped
  // run silently skipped the VLM stage it claimed to scope.
  it('forwards run_vlm=true when set', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await startAutoLabel({ class_id: 7, run_vlm: true });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('run_vlm=true');
  });

  it('omits run_vlm entirely when unset (byte-identical to the pre-G5 request)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await startAutoLabel({ train_clusters: true, vlm_concurrency: 16 });

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).not.toContain('run_vlm');
  });
});

/**
 * `runVlmOnCluster` (2026-09-24 logic-moves W3) — a single
 * `POST {API_PREFIX}/vlm/label_cluster/{id}[?prompt_pack=]`, replacing the
 * old fetch-200-crops-then-chunk-of-64 loop against
 * `{API_PREFIX}/vlm/label_batch`. No client-side crop selection or
 * chunking remains.
 */
describe('runVlmOnCluster', () => {
  const jobResponse = (overrides: Record<string, unknown> = {}) =>
    new Response(
      JSON.stringify({
        job_id: 'job-vlm-1',
        status: 'running',
        stage: 'vlm',
        processed: 0,
        total: 12,
        started_at: 0,
        finished_at: 0,
        error: null,
        result: {},
        args: {},
        eta_seconds: null,
        elapsed_seconds: 0,
        ...overrides,
      }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    );

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('POSTs {API_PREFIX}/vlm/label_cluster/{id} with no crop-fetch round trip', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    const job = await runVlmOnCluster(42);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${API_PREFIX}/vlm/label_cluster/42`);
    expect(init.method).toBe('POST');
    expect(job.job_id).toBe('job-vlm-1');
  });

  it('forwards prompt_pack when provided', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await runVlmOnCluster(42, 'generic_item_v1');

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toContain('prompt_pack=generic_item_v1');
  });

  it('omits prompt_pack entirely when null/undefined', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jobResponse());
    vi.stubGlobal('fetch', fetchMock);

    await runVlmOnCluster(42, null);

    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).not.toContain('prompt_pack');
  });
});

describe('getAutoLabelJobStatus', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns the job state on 200', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          job_id: 'job-1',
          status: 'running',
          stage: 'vlm',
          processed: 1,
          total: 2,
          started_at: 0,
          finished_at: null,
          error: null,
          args: {},
          eta_seconds: null,
          elapsed_seconds: 0,
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const job = await getAutoLabelJobStatus('job-1');

    expect(job?.job_id).toBe('job-1');
    expect(fetchMock.mock.calls[0][0]).toContain('/pipeline/auto_label/status/job-1');
  });

  it('returns null on 404 (unknown job id)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: 'not found' }), {
        status: 404,
        headers: { 'content-type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const job = await getAutoLabelJobStatus('job-unknown');

    expect(job).toBeNull();
  });

  it('rethrows a non-404 error', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: 'boom' }), {
        status: 500,
        headers: { 'content-type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(getAutoLabelJobStatus('job-1')).rejects.toThrow(ApiError);
  });
});

describe('pollAutoLabelJob', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const statusResponse = (body: Record<string, unknown>) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  it('polls {API_PREFIX}/pipeline/auto_label/status until the job leaves running, calling onUpdate each time', async () => {
    const bodies = [
      { status: 'running', stage: 'vlm', processed: 1, total: 3 },
      { status: 'running', stage: 'vlm', processed: 2, total: 3 },
      { status: 'completed', stage: 'finalize', processed: 3, total: 3, result: {} },
    ];
    const fetchMock = vi.fn().mockImplementation(() =>
      Promise.resolve(
        statusResponse({
          job_id: 'job-vlm-1',
          started_at: 0,
          finished_at: 0,
          error: null,
          args: {},
          eta_seconds: null,
          elapsed_seconds: 0,
          ...bodies.shift(),
        }),
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const updates: string[] = [];
    const final = await pollAutoLabelJob((j) => updates.push(j.status), undefined, 0);

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(updates).toEqual(['running', 'running', 'completed']);
    expect(final.status).toBe('completed');
  });

  it('returns immediately (one fetch) when the first poll is already terminal', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      statusResponse({
        job_id: 'job-vlm-1',
        status: 'failed',
        stage: 'vlm',
        processed: 0,
        total: 0,
        started_at: 0,
        finished_at: 0,
        error: 'boom',
        result: {},
        args: {},
        eta_seconds: null,
        elapsed_seconds: 0,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const final = await pollAutoLabelJob(() => {}, undefined, 0);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(final.status).toBe('failed');
    expect(final.error).toBe('boom');
  });

  /**
   * M7 (docs/design/interactive-pass-2026-09-24.md): with `expectedJobId`,
   * polls `GET {API_PREFIX}/pipeline/auto_label/status/{job_id}` — the
   * per-job endpoint the backend now serves (07cc061) — not the
   * "current/most recent job" `.../status` slot, and no longer needs to
   * skip a mismatched `job_id` client-side.
   */
  it('with expectedJobId, polls the per-job status/{job_id} endpoint directly', async () => {
    const bodies = [
      { status: 'running', stage: 'vlm', processed: 1, total: 3 },
      { status: 'completed', stage: 'finalize', processed: 3, total: 3, result: {} },
    ];
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      expect(url).toContain('/pipeline/auto_label/status/job-vlm-2');
      return Promise.resolve(
        statusResponse({
          job_id: 'job-vlm-2',
          started_at: 0,
          finished_at: 0,
          error: null,
          args: {},
          eta_seconds: null,
          elapsed_seconds: 0,
          ...bodies.shift(),
        }),
      );
    });
    vi.stubGlobal('fetch', fetchMock);

    const final = await pollAutoLabelJob(() => {}, undefined, 0, 'job-vlm-2');

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(final.status).toBe('completed');
  });

  it('with expectedJobId, treats a 404 as still-waiting and keeps polling until maxWaitMs', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: 'not found' }), {
        status: 404,
        headers: { 'content-type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(
      pollAutoLabelJob(() => {}, undefined, 0, 'job-never-appears', 5),
    ).rejects.toThrow(/Timed out waiting for job job-never-appears/);
    expect(fetchMock.mock.calls.length).toBeGreaterThan(1);
  });
});

/**
 * `getClusters`/`getCluster` map the served `purity_tier`, `promotable`
 * and `core_similarity_min` (2026-09-24 logic-moves W6) — no client 0.8
 * "pure" threshold, no recomputation.
 */
describe('getClusters purity_tier/promotable/core_similarity_min', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('maps purity_tier, promotable and core_similarity_min from the response verbatim', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [
          {
            cluster_id: 67,
            cluster_kind: 'class',
            size: 37,
            validated_count: 30,
            labelled_count: 37,
            dominant_class_id: 67,
            dominant_class_name: 'widget_b',
            dominant_count: 37,
            purity: 1.0,
            purity_tier: 'pure',
            promotable: true,
            is_unlabeled: false,
            n_subclusters: 0,
            updated_at: null,
            representatives: [],
          },
          {
            cluster_id: 10000,
            cluster_kind: 'candidate',
            size: 108,
            validated_count: 0,
            labelled_count: 3,
            dominant_class_id: null,
            dominant_class_name: null,
            dominant_count: 1,
            purity: 0.33,
            purity_tier: 'noisy',
            promotable: false,
            is_unlabeled: false,
            n_subclusters: 0,
            updated_at: null,
            representatives: [],
          },
        ],
        total: 2,
        total_class_clusters: 1,
        total_candidate_clusters: 1,
        cluster_id_offset: 10000,
        purity_thresholds: {
          pure_min: 0.85,
          mixed_min: 0.6,
          promote_min_members: 4,
          promote_min_labelled_share: 0.5,
        },
        core_similarity_min: 0.75,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getClusters();

    expect(res.items[0]).toMatchObject({
      id: 67,
      purity_tier: 'pure',
      promotable: true,
      core_similarity_min: 0.75,
    });
    expect(res.items[1]).toMatchObject({
      id: 10000,
      purity_tier: 'noisy',
      promotable: false,
      core_similarity_min: 0.75,
    });
    // m7 (2026-09-24 interactive pass): the /clusters legend used to
    // hardcode "≥80% / ≥60%" instead of reading this.
    expect(res.purity_thresholds).toMatchObject({ pure_min: 0.85, mixed_min: 0.6 });
  });

  it('C1: dominant_pct is the served label_purity, never the geometry purity', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          items: [
            {
              cluster_id: 64,
              cluster_kind: 'class',
              size: 616,
              validated_count: 0,
              labelled_count: 616,
              dominant_class_id: 64,
              dominant_class_name: 'class_b',
              dominant_count: 616,
              purity: 0.03,
              purity_n: 616,
              purity_basis: 'nearest_centroid',
              purity_tier: 'noisy',
              label_purity: 1.0,
              labelled_share: 1.0,
              promotable: false,
              is_unlabeled: false,
              n_subclusters: 0,
              updated_at: null,
              representatives: [],
            },
          ],
          total: 1,
          total_class_clusters: 1,
          total_candidate_clusters: 0,
          cluster_id_offset: 10000,
        }),
      ),
    );
    const res = await getClusters();
    expect(res.items[0]).toMatchObject({
      dominant_pct: 1.0,
      purity: 0.03,
      dominant_count: 616,
      labelled_count: 616,
    });
  });

  it('getClusters reports purity_thresholds as null when the server omits it, never a stale hardcoded value', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [],
        total: 0,
        total_class_clusters: 0,
        total_candidate_clusters: 0,
        cluster_id_offset: 0,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const res = await getClusters();
    expect(res.purity_thresholds ?? null).toBeNull();
  });

  it('getCluster carries purity_tier/promotable/core_similarity_min through for the single-cluster lookup', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.startsWith(`${API_PREFIX}/crops`)) {
        return Promise.resolve(
          jsonResponse({ total: 0, page: 1, page_size: 60, crops: [] }),
        );
      }
      return Promise.resolve(
        jsonResponse({
          items: [
            {
              cluster_id: 67,
              cluster_kind: 'class',
              size: 37,
              validated_count: 30,
              labelled_count: 37,
              dominant_class_id: 67,
              dominant_class_name: 'widget_b',
              dominant_count: 37,
              purity: 1.0,
              purity_tier: 'pure',
              promotable: true,
              is_unlabeled: false,
              n_subclusters: 0,
              updated_at: null,
              representatives: [],
            },
          ],
          total: 1,
          total_class_clusters: 1,
          total_candidate_clusters: 0,
          cluster_id_offset: 10000,
          core_similarity_min: 0.75,
        }),
      );
    });
    vi.stubGlobal('fetch', fetchMock);

    const res = await getCluster(67, 1, 60);

    expect(res.cluster).toMatchObject({
      id: 67,
      cluster_kind: 'class',
      purity_tier: 'pure',
      promotable: true,
      core_similarity_min: 0.75,
    });
  });

  it('getCluster falls back to a null-identity stub (purity_tier null, promotable false) when the cluster-card lookup fails', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.startsWith(`${API_PREFIX}/crops`)) {
        return Promise.resolve(
          jsonResponse({ total: 0, page: 1, page_size: 60, crops: [] }),
        );
      }
      return Promise.reject(new Error('network blip'));
    });
    vi.stubGlobal('fetch', fetchMock);

    const res = await getCluster(999, 1, 60);

    expect(res.cluster.purity_tier).toBeNull();
    expect(res.cluster.promotable).toBe(false);
    expect(res.cluster.core_similarity_min).toBeNull();
  });
});

/**
 * DQ-M2 fix (dq-queues cutover, 2026-09-24): `purity` is now
 * nearest-centroid geometry purity (`purity_basis: 'nearest_centroid'`,
 * `purity_n` members measured), independent of `label_purity`
 * (the old label-based number, always 1.0 for a class cluster) and
 * `labelled_share`. `getClusters` must map all four verbatim.
 */
describe('getClusters purity_n/purity_basis/label_purity/labelled_share (DQ-M2)', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('maps purity_n/purity_basis/label_purity/labelled_share verbatim', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [
          {
            cluster_id: 43,
            cluster_kind: 'class',
            size: 24,
            validated_count: 0,
            labelled_count: 24,
            dominant_class_id: 43,
            dominant_class_name: 'motardbike',
            dominant_count: 24,
            purity: 0.29,
            purity_n: 24,
            purity_basis: 'nearest_centroid',
            purity_tier: 'noisy',
            label_purity: 1.0,
            labelled_share: 1.0,
            promotable: false,
            is_unlabeled: false,
            n_subclusters: 0,
            updated_at: null,
            representatives: [],
          },
        ],
        total: 1,
        total_class_clusters: 1,
        total_candidate_clusters: 0,
        cluster_id_offset: 10000,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getClusters();

    expect(res.items[0]).toMatchObject({
      id: 43,
      purity: 0.29,
      purity_n: 24,
      purity_basis: 'nearest_centroid',
      label_purity: 1.0,
      labelled_share: 1.0,
    });
    // The whole point of DQ-M2: purity and label_purity must be able to
    // disagree (a class cluster is no longer 1.0-purity by construction).
    expect(res.items[0].purity).not.toBe(res.items[0].label_purity);
  });

  it('leaves the new fields null when the server omits them (older backend)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [
          {
            cluster_id: 67,
            cluster_kind: 'class',
            size: 10,
            validated_count: 0,
            labelled_count: 10,
            dominant_class_id: 67,
            dominant_class_name: 'widget_b',
            dominant_count: 10,
            purity: 1.0,
            purity_tier: 'pure',
            promotable: false,
            is_unlabeled: false,
            n_subclusters: 0,
            updated_at: null,
            representatives: [],
          },
        ],
        total: 1,
        total_class_clusters: 1,
        total_candidate_clusters: 0,
        cluster_id_offset: 10000,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getClusters();
    expect(res.items[0].purity_n ?? null).toBeNull();
    expect(res.items[0].purity_basis ?? null).toBeNull();
    expect(res.items[0].label_purity ?? null).toBeNull();
    expect(res.items[0].labelled_share ?? null).toBeNull();
  });
});

/**
 * D-4 (docs/design/curation_query_performance_audit.md): `/clusters`
 * representatives are paged by offset/limit independent of the card list
 * itself. `getClusters` forwards `representatives_offset`/
 * `representatives_limit` as `offset`/`limit` query params and echoes the
 * response's window back so a caller can advance past exactly what it got.
 */
describe('getClusters D-4 representatives windowing', () => {
  const jsonResponse = (body: unknown) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('sends representatives_offset/representatives_limit as offset/limit', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [],
        total: 0,
        total_class_clusters: 0,
        total_candidate_clusters: 0,
        cluster_id_offset: 10000,
        representatives_offset: 24,
        representatives_limit: 24,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await getClusters({ representatives_offset: 24, representatives_limit: 24 });

    const calledUrl = fetchMock.mock.calls[0][0] as string;
    expect(calledUrl).toContain('offset=24');
    expect(calledUrl).toContain('limit=24');
  });

  it('echoes the response window back on ClustersResponse, not the requested one', async () => {
    // The backend can serve a smaller window than requested (e.g. near the
    // end of the list) — the caller must advance by what actually came
    // back, not by what it asked for, or it'll skip/repeat cards.
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [],
        total: 30,
        total_class_clusters: 30,
        total_candidate_clusters: 0,
        cluster_id_offset: 10000,
        representatives_offset: 24,
        representatives_limit: 6,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await getClusters({
      representatives_offset: 24,
      representatives_limit: 24,
    });

    expect(res.representatives_offset).toBe(24);
    expect(res.representatives_limit).toBe(6);
  });

  it('omits offset/limit query params when no window is requested', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [],
        total: 0,
        total_class_clusters: 0,
        total_candidate_clusters: 0,
        cluster_id_offset: 10000,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await getClusters({});

    const calledUrl = fetchMock.mock.calls[0][0] as string;
    expect(calledUrl).not.toContain('offset=');
    expect(calledUrl).not.toContain('limit=');
  });
});

describe('API_PREFIX', () => {
  // T-E2: the default matches OpenProcessor's OP_API_PREFIX default.
  it('defaults to /curation when PUBLIC_API_PREFIX is unset', () => {
    expect(API_PREFIX).toBe('/curation');
  });

  it('treats empty, whitespace and an unsubstituted placeholder as unset', () => {
    expect(normalizeApiPrefix('')).toBe('/curation');
    expect(normalizeApiPrefix('   ')).toBe('/curation');
    expect(normalizeApiPrefix('__API_PREFIX__')).toBe('/curation');
  });

  it('normalizes a configured prefix to a leading slash and no trailing slash', () => {
    expect(normalizeApiPrefix('/curation')).toBe('/curation');
    expect(normalizeApiPrefix('curation')).toBe('/curation');
    expect(normalizeApiPrefix('/curation/')).toBe('/curation');
    expect(normalizeApiPrefix('/curation///')).toBe('/curation');
  });
});

/**
 * `getCurationSettings`/`putCurationDefaults` — docs/design/
 * curation-settings-ui-plan-2026-09-21.md §4.1. Deliberately asymmetric
 * vs. `getMethods`: THIS endpoint throws on failure rather than
 * degrading to a fallback, since a settings page must tell a 404
 * ("backend predates the feature") apart from every other outcome.
 */
describe('getCurationSettings / putCurationDefaults', () => {
  const jsonResponse = (body: unknown, init: ResponseInit = {}) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
      ...init,
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('getCurationSettings composes ${API_PREFIX}/settings and parses the body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        defaults: { cluster: 'ivf' },
        updated_at: '2026-09-20T23:04:39+00:00',
        updated_by: null,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await getCurationSettings();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const calledUrl = (fetchMock.mock.calls[0]?.[0] as string) ?? '';
    expect(calledUrl).toContain(`${API_PREFIX}/settings`);
    expect(result).toEqual({
      defaults: { cluster: 'ivf' },
      updated_at: '2026-09-20T23:04:39+00:00',
      updated_by: null,
    });
  });

  it('putCurationDefaults sends PUT with the exact partial body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        defaults: { sort: 'uncertainty_entropy' },
        updated_at: '2026-09-21T00:00:00+00:00',
        updated_by: null,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await putCurationDefaults({ sort: 'uncertainty_entropy' });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [calledUrl, calledInit] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(calledUrl).toContain(`${API_PREFIX}/settings`);
    expect(calledInit.method).toBe('PUT');
    expect(calledInit.body).toBe(
      JSON.stringify({ defaults: { sort: 'uncertainty_entropy' } }),
    );
    expect(result.defaults).toEqual({ sort: 'uncertainty_entropy' });
  });

  it('putCurationDefaults accepts null to clear an axis, sending the literal null', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        defaults: {},
        updated_at: '2026-09-21T00:00:00+00:00',
        updated_by: null,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await putCurationDefaults({ sort: null });

    const [, calledInit] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(calledInit.body).toBe(JSON.stringify({ defaults: { sort: null } }));
    expect(result.defaults).toEqual({});
  });

  it('getCurationSettings THROWS on 404 (unlike getMethods, which falls back)', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'not found' }), { status: 404 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await expect(getCurationSettings()).rejects.toBeInstanceOf(ApiError);
    await expect(getCurationSettings()).rejects.toMatchObject({ status: 404 });

    // Contrast: getMethods() on the identical 404 response resolves rather
    // than rejecting. Same fetch mock, different endpoint contract.
    await expect(getMethods()).resolves.toEqual(FALLBACK_METHODS);
  });
});

/**
 * `/scores/*` wrappers (docs/design/frontend-coverage-audit-2026-09-24.md
 * §G10). Wire shapes confirmed against openprocessor `main`'s
 * `crop_scores/job.py::compute_coverage`/`_JobState` — `coverage` is
 * `{scorer_id: {field, n_scored, total, pct}}`, and every job snapshot
 * is `{job_id, status, scorers, processed, total, started_at,
 * finished_at, error, results}`.
 */
describe('getScoresCoverage / computeScores / getScoresStatus / cancelScores', () => {
  const jsonResponse = (body: unknown, init: ResponseInit = {}) =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
      ...init,
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('getScoresCoverage composes ${API_PREFIX}/scores/coverage and parses the coverage map', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        coverage: {
          uniqueness: { field: 'uniqueness_score', n_scored: 0, total: 7961, pct: 0 },
          near_dup: { field: 'dup_group_id', n_scored: 12, total: 7961, pct: 0.15 },
          mistakenness: {
            field: 'mistakenness_score',
            n_scored: 3200,
            total: 7961,
            pct: 40.2,
          },
        },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await getScoresCoverage();

    const calledUrl = (fetchMock.mock.calls[0]?.[0] as string) ?? '';
    expect(calledUrl).toContain(`${API_PREFIX}/scores/coverage`);
    expect(Object.keys(result)).toEqual(['uniqueness', 'near_dup', 'mistakenness']);
    expect(result.mistakenness).toEqual({
      field: 'mistakenness_score',
      n_scored: 3200,
      total: 7961,
      pct: 40.2,
    });
  });

  it('getScoresCoverage drops a malformed entry instead of crashing the whole parse', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        coverage: {
          uniqueness: { field: 'uniqueness_score', n_scored: 5, total: 10, pct: 50 },
          broken: { field: 'x' }, // missing n_scored/total/pct
          alsoBroken: null,
        },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await getScoresCoverage();
    expect(Object.keys(result)).toEqual(['uniqueness']);
  });

  it('getScoresCoverage THROWS on 404 — the caller (ScoresCard) uses this to hide the card, not to fall back', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'not found' }), { status: 404 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await expect(getScoresCoverage()).rejects.toBeInstanceOf(ApiError);
    await expect(getScoresCoverage()).rejects.toMatchObject({ status: 404 });
  });

  it('computeScores POSTs {scorers: null} verbatim for "compute all"', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        job_id: 'abc',
        status: 'running',
        scorers: ['uniqueness', 'near_dup', 'mistakenness'],
        processed: 0,
        total: 7961,
        started_at: 1000,
        finished_at: 0,
        error: null,
        results: {},
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const job = await computeScores(null);

    const [calledUrl, calledInit] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(calledUrl).toContain(`${API_PREFIX}/scores/compute`);
    expect(calledInit.method).toBe('POST');
    expect(calledInit.body).toBe(JSON.stringify({ scorers: null }));
    expect(job.status).toBe('running');
    expect(job.scorers).toEqual(['uniqueness', 'near_dup', 'mistakenness']);
    expect(job.total).toBe(7961);
  });

  it('computeScores POSTs the exact selected scorer ids, not a re-derived list', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        job_id: 'abc',
        status: 'running',
        scorers: ['near_dup'],
        processed: 0,
        total: 100,
        started_at: 1000,
        finished_at: 0,
        error: null,
        results: {},
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await computeScores(['near_dup']);

    const [, calledInit] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(calledInit.body).toBe(JSON.stringify({ scorers: ['near_dup'] }));
  });

  it('computeScores rejects with the server detail verbatim (e.g. a scorer lacking its inputs)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          detail:
            "unknown scorer(s): ['bogus']; valid: ['mistakenness', 'near_dup', 'uniqueness']",
        }),
        { status: 400 },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(computeScores(['bogus'])).rejects.toMatchObject({
      status: 400,
      detail:
        "unknown scorer(s): ['bogus']; valid: ['mistakenness', 'near_dup', 'uniqueness']",
    });
  });

  it('getScoresStatus parses a failed job, keeping the server error string verbatim', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        job_id: 'abc',
        status: 'failed',
        scorers: ['mistakenness'],
        processed: 0,
        total: 7961,
        started_at: 1000,
        finished_at: 1010,
        error: 'mistakenness requires probe_pred_confidence — none scored yet',
        results: {},
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const job = await getScoresStatus();
    const calledUrl = (fetchMock.mock.calls[0]?.[0] as string) ?? '';
    expect(calledUrl).toContain(`${API_PREFIX}/scores/status`);
    expect(job.status).toBe('failed');
    expect(job.error).toBe(
      'mistakenness requires probe_pred_confidence — none scored yet',
    );
  });

  it('getScoresStatus resolves to an idle snapshot on a malformed body rather than throwing', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('not json', {
        status: 200,
        headers: { 'content-type': 'text/plain' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const job = await getScoresStatus();
    expect(job.status).toBe('idle');
    expect(job.scorers).toEqual([]);
  });

  it('cancelScores POSTs to /scores/cancel and reports the served cancelled flag', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        cancelled: true,
        job_id: 'abc',
        status: 'cancelled',
        scorers: ['uniqueness'],
        processed: 3,
        total: 7961,
        started_at: 1000,
        finished_at: 1005,
        error: null,
        results: {},
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await cancelScores();
    const [calledUrl, calledInit] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(calledUrl).toContain(`${API_PREFIX}/scores/cancel`);
    expect(calledInit.method).toBe('POST');
    expect(result.cancelled).toBe(true);
    expect(result.status).toBe('cancelled');
  });

  it('cancelScores reports cancelled:false when nothing was running', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ cancelled: false, status: 'idle' }));
    vi.stubGlobal('fetch', fetchMock);

    const result = await cancelScores();
    expect(result.cancelled).toBe(false);
  });
});
