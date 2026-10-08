/**
 * P3 lifecycle wrappers (every one GLOBAL, never scoped), the served
 * error parsing, and `apiFetch`'s stale-project guard: a scoped response
 * that lands after the active project changed is dropped as an
 * AbortError, so the previous project's data never renders.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  API_PREFIX,
  ApiError,
  archiveProject,
  cloneProjectSettings,
  createProject,
  deleteProject,
  deleteProjectDryRun,
  getHealth,
  getProjects,
  patchProject,
  projectErrorDetail,
  apiErrorText,
  scopeGeneration,
  setScopedPrefix,
  unarchiveProject,
} from './api';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  setScopedPrefix(API_PREFIX);
});

function capture(body: unknown = { project: {}, warnings: [] }, status = 200) {
  const fetchMock = vi.fn().mockResolvedValue(json(body, status));
  vi.stubGlobal('fetch', fetchMock);
  return () => {
    const [url, init] = fetchMock.mock.calls[0]! as [string, RequestInit];
    return {
      url: String(url),
      method: init.method ?? 'GET',
      body: init.body ? JSON.parse(String(init.body)) : undefined,
    };
  };
}

describe('lifecycle wrappers hit the global /projects routes', () => {
  it('create: POST /projects with the request body', async () => {
    const sent = capture();
    await createProject({ slug: 'alpha', display_name: 'Alpha', description: 'd' });
    expect(sent()).toEqual({
      url: `${API_PREFIX}/projects`,
      method: 'POST',
      body: { slug: 'alpha', display_name: 'Alpha', description: 'd' },
    });
  });

  it('patch: PATCH /projects/{slug} with expected_revision', async () => {
    const sent = capture();
    await patchProject('alpha', { display_name: 'A2', expected_revision: 4 });
    expect(sent()).toEqual({
      url: `${API_PREFIX}/projects/alpha`,
      method: 'PATCH',
      body: { display_name: 'A2', expected_revision: 4 },
    });
  });

  it('archive / unarchive: POST with expected_revision', async () => {
    let sent = capture();
    await archiveProject('alpha', { expected_revision: 4 });
    expect(sent()).toEqual({
      url: `${API_PREFIX}/projects/alpha/archive`,
      method: 'POST',
      body: { expected_revision: 4 },
    });
    sent = capture();
    await unarchiveProject('alpha', { expected_revision: 5 });
    expect(sent()).toMatchObject({
      url: `${API_PREFIX}/projects/alpha/unarchive`,
      method: 'POST',
    });
  });

  it('clone_settings: POST {from, axes, expected_revision}', async () => {
    const sent = capture();
    await cloneProjectSettings('alpha', {
      from: 'default',
      axes: ['classes'],
      expected_revision: 4,
    });
    expect(sent()).toEqual({
      url: `${API_PREFIX}/projects/alpha/clone_settings`,
      method: 'POST',
      body: { from: 'default', axes: ['classes'], expected_revision: 4 },
    });
  });

  it('delete: dry run and confirm are DELETE query params', async () => {
    let sent = capture({ blocking: [] });
    await deleteProjectDryRun('alpha');
    expect(sent()).toMatchObject({
      url: `${API_PREFIX}/projects/alpha?dry_run=true`,
      method: 'DELETE',
    });
    sent = capture();
    await deleteProject('alpha', 'alpha');
    expect(sent()).toMatchObject({
      url: `${API_PREFIX}/projects/alpha?confirm=alpha`,
      method: 'DELETE',
    });
  });

  it('getProjects sends include_archived only when asked', async () => {
    let sent = capture({ projects: [] });
    await getProjects();
    expect(sent().url).toBe(`${API_PREFIX}/projects`);
    sent = capture({ projects: [] });
    await getProjects(undefined, true);
    expect(sent().url).toBe(`${API_PREFIX}/projects?include_archived=true`);
  });
});

describe('projectErrorDetail / apiErrorText', () => {
  const conflict = {
    detail: {
      error: 'revision_conflict',
      message: 'expected revision 4, current is 6',
      project: 'alpha',
      current_revision: 6,
    },
  };

  it('reads the structured detail and shows its message verbatim', () => {
    const e = new ApiError(409, 'u', conflict);
    expect(projectErrorDetail(e)).toMatchObject({
      error: 'revision_conflict',
      current_revision: 6,
    });
    expect(apiErrorText(e)).toBe('expected revision 4, current is 6');
  });

  it('falls back to the generic detail for a pydantic 422 list', () => {
    const e = new ApiError(422, 'u', {
      detail: [{ loc: ['body', 'display_name'], msg: 'Field required', type: 'missing' }],
    });
    expect(projectErrorDetail(e)).toBeNull();
    expect(apiErrorText(e)).toContain('display_name');
  });

  it('is null for a non-ApiError', () => {
    expect(projectErrorDetail(new Error('boom'))).toBeNull();
    expect(apiErrorText(new Error('boom'))).toBe('boom');
  });
});

describe('stale-project guard', () => {
  it('drops a scoped response that lands after the project changed', async () => {
    let release!: (r: Response) => void;
    vi.stubGlobal(
      'fetch',
      vi.fn().mockReturnValue(new Promise<Response>((r) => (release = r))),
    );
    setScopedPrefix(`${API_PREFIX}/projects/alpha`);
    const pending = getHealth();
    setScopedPrefix(`${API_PREFIX}/projects/beta`);
    release(json({ status: 'ok', region_profile: { name: 'alpha_only' } }));
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('keeps a scoped response when the project did not change', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({ status: 'ok' })));
    setScopedPrefix(`${API_PREFIX}/projects/alpha`);
    await expect(getHealth()).resolves.toEqual({ status: 'ok' });
  });

  it('never drops a GLOBAL response', async () => {
    let release!: (r: Response) => void;
    vi.stubGlobal(
      'fetch',
      vi.fn().mockReturnValue(new Promise<Response>((r) => (release = r))),
    );
    setScopedPrefix(`${API_PREFIX}/projects/alpha`);
    const pending = archiveProject('alpha', { expected_revision: 1 });
    setScopedPrefix(`${API_PREFIX}/projects/beta`);
    release(json({ project: { slug: 'alpha' } }));
    await expect(pending).resolves.toEqual({ project: { slug: 'alpha' } });
  });

  it('re-selecting the same prefix is not a change', () => {
    setScopedPrefix(`${API_PREFIX}/projects/alpha`);
    const gen = scopeGeneration();
    setScopedPrefix(`${API_PREFIX}/projects/alpha`);
    expect(scopeGeneration()).toBe(gen);
  });
});
