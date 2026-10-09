/**
 * `api_combine.ts`: every wrapper's method, URL (global vs the target /
 * source project's own served prefix) and body.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import {
  cancelCombine,
  getCombineJob,
  getDatasetFormatsFor,
  isCombineNotFound,
  previewCombine,
  resumeCombine,
  runCombineNextStep,
  startCombine,
} from '$lib/api_combine';
import { ApiError } from '$lib/api';
import type { CombineRequest } from '$lib/types_combine';

const REQ: CombineRequest = {
  target: { slug: 'merged', display_name: 'Merged' },
  sources: [{ project: 'widgets-a' }, { project: 'widgets-b' }],
};

function stub(body: unknown = {}, status = 200) {
  const fetchMock = vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), {
      status,
      headers: { 'content-type': 'application/json' },
    }),
  );
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}
const call = (m: ReturnType<typeof vi.fn>) => ({
  url: String(m.mock.calls[0]![0]),
  init: m.mock.calls[0]![1] as RequestInit,
});

afterEach(() => vi.unstubAllGlobals());

describe('combine wrappers', () => {
  it('preview POSTs the request to the global route', async () => {
    const m = stub({ ok: true });
    await previewCombine(REQ);
    const { url, init } = call(m);
    expect(url).toBe(`${API_PREFIX}/projects/combine/preview`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(String(init.body))).toEqual(REQ);
  });

  it('start POSTs the request plus expected_preview_sha, with no project key', async () => {
    const m = stub({ job_id: 'cmb_1', target: 'merged' }, 202);
    await startCombine({ ...REQ, expected_preview_sha: 'abc' });
    const { url, init } = call(m);
    expect(url).toBe(`${API_PREFIX}/projects/combine`);
    expect(init.method).toBe('POST');
    const body = JSON.parse(String(init.body));
    expect(body.expected_preview_sha).toBe('abc');
    expect(body).not.toHaveProperty('project');
  });

  it('job read, cancel and resume encode the id and use the right verbs', async () => {
    const m1 = stub({ job_id: 'a/b', status: 'running' });
    await getCombineJob('a/b');
    expect(call(m1).url).toBe(`${API_PREFIX}/projects/combine/a%2Fb`);
    expect(call(m1).init.method).toBeUndefined();

    const m2 = stub({ job_id: 'j', status: 'running' });
    await cancelCombine('a/b');
    expect(call(m2).url).toBe(`${API_PREFIX}/projects/combine/a%2Fb/cancel`);
    expect(call(m2).init.method).toBe('POST');

    const m3 = stub({ job_id: 'j', status: 'queued' });
    await resumeCombine('a/b');
    expect(call(m3).url).toBe(`${API_PREFIX}/projects/combine/a%2Fb/resume`);
    expect(call(m3).init.method).toBe('POST');
  });

  it('the formats read and the next step go through the named project prefix', async () => {
    const m1 = stub({ mapping_actions: [] });
    await getDatasetFormatsFor({ prefix: '/curation/projects/widgets-a' });
    expect(call(m1).url).toBe('/curation/projects/widgets-a/datasets/formats');

    const m2 = stub({});
    await runCombineNextStep(
      { prefix: '/curation/projects/merged' },
      { action: 'recluster', method: 'post', path: '/cluster/umap/rebuild', reason: 'x' },
    );
    const { url, init } = call(m2);
    expect(url).toBe('/curation/projects/merged/cluster/umap/rebuild');
    expect(init.method).toBe('POST');
    expect(init.body).toBeUndefined();
  });
});

describe('isCombineNotFound', () => {
  it('is true only for a 404 carrying the structured combine_not_found detail', () => {
    const structured = new ApiError(404, '/u', {
      detail: { error: 'combine_not_found', message: 'no combine job' },
    });
    expect(isCombineNotFound(structured)).toBe(true);
    expect(isCombineNotFound(new ApiError(404, '/u', { detail: 'Not Found' }))).toBe(
      false,
    );
    expect(
      isCombineNotFound(
        new ApiError(409, '/u', { detail: { error: 'combine_not_found', message: '' } }),
      ),
    ).toBe(false);
    expect(isCombineNotFound(new Error('x'))).toBe(false);
  });
});
