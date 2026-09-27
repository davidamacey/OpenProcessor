/**
 * The route loads that carry the active project in the URL path:
 * `/` and the legacy bare paths redirect under `/p/<served default>`
 * with the query string kept, and `/p/[project]`'s layout resolves the
 * slug — selecting it (moving `scoped()`) only when the server lists it
 * as selectable.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { isHttpError, isRedirect } from '@sveltejs/kit';
import { API_PREFIX, scoped } from '$lib/api';
import { projectsStore } from '$stores/projects.svelte';
import { testProject } from '$lib/test/fixtures/projects';
import { load as rootLoad } from './+page';
import { load as legacyLoad } from './[...legacy]/+page';
import { load as projectLayoutLoad } from './p/[project]/+layout';

const DEFAULT = testProject({ slug: 'default', prefix: API_PREFIX, is_default: true });
const ALPHA = testProject({ slug: 'alpha' });
const WIP = testProject({
  slug: 'wip',
  status: 'building',
  selectable: false,
  writable: false,
});

/** Runs a load and returns where it redirected, or the thrown error. */
async function outcome(
  run: () => unknown,
): Promise<{ location?: string; status?: number }> {
  try {
    await run();
  } catch (e) {
    if (isRedirect(e)) return { location: e.location, status: e.status };
    if (isHttpError(e)) return { status: e.status };
    throw e;
  }
  return {};
}

const parent = async () => ({ apiBase: '', projectsError: null });

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const call = (fn: any, event: Record<string, unknown>) => fn({ parent, ...event });

beforeEach(() => {
  projectsStore.list = [DEFAULT, ALPHA, WIP];
  projectsStore.defaultSlug = 'default';
});

afterEach(() => {
  vi.unstubAllGlobals();
  projectsStore.select(DEFAULT);
});

describe('/ and the legacy bare paths', () => {
  it('/ goes to the served default project, query kept', async () => {
    const r = await outcome(() => call(rootLoad, { url: new URL('http://x/?a=1') }));
    expect(r).toEqual({ location: '/p/default/dashboard?a=1', status: 307 });
  });

  it('follows the served default_slug, not a hardcoded one', async () => {
    projectsStore.defaultSlug = 'alpha';
    const r = await outcome(() => call(rootLoad, { url: new URL('http://x/') }));
    expect(r.location).toBe('/p/alpha/dashboard');
  });

  it('a bare section path keeps its sub-path and query', async () => {
    const r = await outcome(() =>
      call(legacyLoad, { url: new URL('http://x/review?tab=regions&crop_id=c1') }),
    );
    expect(r).toEqual({
      location: '/p/default/review?tab=regions&crop_id=c1',
      status: 307,
    });
    const nested = await outcome(() =>
      call(legacyLoad, { url: new URL('http://x/clusters/12') }),
    );
    expect(nested.location).toBe('/p/default/clusters/12');
  });

  it('anything else is a 404', async () => {
    const r = await outcome(() => call(legacyLoad, { url: new URL('http://x/nope') }));
    expect(r).toEqual({ status: 404 });
  });

  it('with nothing selectable, both go to the project list', async () => {
    projectsStore.list = [WIP];
    expect(
      (await outcome(() => call(rootLoad, { url: new URL('http://x/') }))).location,
    ).toBe('/projects');
    expect(
      (await outcome(() => call(legacyLoad, { url: new URL('http://x/review') })))
        .location,
    ).toBe('/projects');
  });
});

describe('/p/[project] layout', () => {
  function stubScopedBoot(): ReturnType<typeof vi.fn> {
    const fetchMock = vi.fn(async (url: string) => {
      if (String(url).endsWith('/health')) {
        return new Response(JSON.stringify({ status: 'ok', region_profile: null }), {
          headers: { 'content-type': 'application/json' },
        });
      }
      return new Response('{"detail":"not found"}', {
        status: 404,
        headers: { 'content-type': 'application/json' },
      });
    });
    vi.stubGlobal('fetch', fetchMock);
    return fetchMock;
  }

  it('selects a listed selectable slug and boots it on its served prefix', async () => {
    const fetchMock = stubScopedBoot();
    const data = await call(projectLayoutLoad, { params: { project: 'alpha' } });
    expect(data.resolution).toMatchObject({ kind: 'ok', project: { slug: 'alpha' } });
    expect(projectsStore.current?.slug).toBe('alpha');
    expect(scoped()).toBe(ALPHA.prefix);
    const urls = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(urls).toContain(`${ALPHA.prefix}/health`);
    expect(urls.every((u) => u.startsWith(ALPHA.prefix))).toBe(true);
  });

  it('a served non-selectable slug is "unavailable" and never becomes active', async () => {
    const fetchMock = stubScopedBoot();
    const data = await call(projectLayoutLoad, { params: { project: 'wip' } });
    expect(data.resolution).toMatchObject({
      kind: 'unavailable',
      project: { slug: 'wip' },
    });
    expect(projectsStore.current?.slug).toBe('default');
    expect(scoped()).toBe(API_PREFIX);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('an unknown slug is "not found" (after one GET /projects/{slug}) and fires nothing scoped', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ detail: { error: 'project_not_found', message: 'no' } }),
        {
          status: 404,
          headers: { 'content-type': 'application/json' },
        },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);
    const data = await call(projectLayoutLoad, { params: { project: 'ghost' } });
    expect(data.resolution).toEqual({ kind: 'not_found', slug: 'ghost' });
    expect(fetchMock.mock.calls.map((c) => String(c[0]))).toEqual([
      `${API_PREFIX}/projects/ghost`,
    ]);
    expect(scoped()).toBe(API_PREFIX);
  });
});
