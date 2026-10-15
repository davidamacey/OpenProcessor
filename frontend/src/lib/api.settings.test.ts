/**
 * Tests for the settings-page wrappers: `/methods`, the shared curation
 * defaults and the curation-scores endpoints.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { ApiError } from '$lib/api_errors';
import {
  cancelScores,
  computeScores,
  getScoresCoverage,
  getScoresStatus,
} from '$lib/api_scores';
import { getCurationSettings, getMethods, putCurationDefaults } from '$lib/api_settings';

/**
 * getMethods() rejects on failure like every other read — there is no
 * hardcoded capability list to fall back to.
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
  });

  it('rejects with the ApiError on a 404, without retrying', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'not found' }), { status: 404 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await expect(getMethods()).rejects.toMatchObject({ status: 404 });
    // No retry on 4xx — matches apiFetch's documented "don't retry on 4xx" rule.
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('rejects on a network failure, after the normal 5xx/network retry budget', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError('fetch failed'));
    vi.stubGlobal('fetch', fetchMock);

    await expect(getMethods()).rejects.toThrow();
    // apiFetch's retry loop: 1 initial + 3 retries = 4 attempts.
    expect(fetchMock).toHaveBeenCalledTimes(4);
  }, 10_000);

  it('parses a malformed 200 body to empty lists (not a crash)', async () => {
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
      vlm: [],
      axes: [],
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
        resource_links: [],
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
      resource_links: [],
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

  it('getCurationSettings throws on 404', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'not found' }), { status: 404 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await expect(getCurationSettings()).rejects.toBeInstanceOf(ApiError);
    await expect(getCurationSettings()).rejects.toMatchObject({ status: 404 });
  });
});

/**
 * `/scores/*` wrappers (docs/design/frontend-coverage-audit-2026-09-24.md
 * §G10). Wire shapes confirmed against OpenProcessor `main`'s
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
