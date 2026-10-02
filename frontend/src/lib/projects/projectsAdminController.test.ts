/**
 * `/projects` controller: every lifecycle write sends exactly what the
 * served API asks for, the envelope's `warnings` become toasts, and
 * every refusal comes back as the served `{error, message}` — message
 * verbatim, code for the page to pick its follow-up.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { createProjectsAdmin } from '$lib/projects/projectsAdminController.svelte';
import { projectsStore } from '$stores/projects.svelte';
import { toastStore } from '$stores/toast.svelte';
import { testProject, testProjectsResponse } from '$lib/test/fixtures/projects';

const DEFAULT = testProject({
  slug: 'default',
  prefix: API_PREFIX,
  is_default: true,
  deletable: false,
});
const ALPHA = testProject({ slug: 'alpha', revision: 7 });

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

type Route = (url: string, init: RequestInit) => Response | undefined;
let calls: { method: string; url: string; body: unknown }[] = [];

function serve(route: Route): void {
  calls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const method = init.method ?? 'GET';
      calls.push({
        method,
        url: String(url),
        body: init.body ? JSON.parse(String(init.body)) : undefined,
      });
      if (method === 'GET' && /\/projects(\?|$)/.test(String(url))) {
        return json(testProjectsResponse([DEFAULT, ALPHA]));
      }
      return route(String(url), init) ?? json({ detail: 'unrouted' }, 500);
    }),
  );
}

const writes = () => calls.filter((c) => c.method !== 'GET');

beforeEach(() => {
  toastStore.toasts = [];
});

afterEach(() => {
  vi.unstubAllGlobals();
  projectsStore.select(DEFAULT);
});

describe('load', () => {
  it('reads the served list, capacity and limits; include_archived only when toggled', async () => {
    serve(() => undefined);
    const admin = createProjectsAdmin();
    await admin.load();
    expect(admin.list.map((p) => p.slug)).toEqual(['default', 'alpha']);
    expect(admin.capacity?.status).toBe('ok');
    expect(admin.limits?.cloneable_axes).toEqual(['settings_defaults', 'classes']);
    expect(calls[0]!.url).toBe(`${API_PREFIX}/projects`);
    await admin.setIncludeArchived(true);
    expect(calls.at(-1)!.url).toBe(`${API_PREFIX}/projects?include_archived=true`);
  });
});

describe('create', () => {
  it('posts the form and turns every served warning into its own toast', async () => {
    const created = testProject({ slug: 'beta' });
    serve((url, init) =>
      init.method === 'POST'
        ? json(
            {
              project: created,
              warnings: [
                { code: 'shard_budget_high', message: 'served: near the shard budget' },
              ],
            },
            201,
          )
        : undefined,
    );
    const admin = createProjectsAdmin();
    const res = await admin.create({
      slug: 'beta',
      display_name: 'Beta',
      description: '',
    });
    expect(res).toMatchObject({ ok: true, project: { slug: 'beta' } });
    expect(writes()).toEqual([
      {
        method: 'POST',
        url: `${API_PREFIX}/projects`,
        body: { slug: 'beta', display_name: 'Beta', description: '' },
      },
    ]);
    expect(toastStore.toasts.map((t) => [t.kind, t.text])).toContainEqual([
      'warn',
      'served: near the shard budget',
    ]);
    // Both this page's list and the switcher's are re-read from the server.
    expect(calls.filter((c) => c.method === 'GET').map((c) => c.url)).toEqual([
      `${API_PREFIX}/projects`,
      `${API_PREFIX}/projects`,
    ]);
  });

  it('lists every keymap action the clone axis dropped, from the served keymap_clone_conflicts', async () => {
    serve((url, init) =>
      init.method === 'POST'
        ? json({
            project: ALPHA,
            keymap_clone_conflicts: [
              {
                action_id: 'review.queue.discard',
                combo: 'd',
                class_id: 4,
                class_name: 'dog',
              },
              { action_id: 'cluster.undo', combo: 'z', class_id: 9, class_name: 'zebra' },
            ],
          })
        : undefined,
    );
    const res = await createProjectsAdmin().cloneSettings(ALPHA, 'default', ['keymap']);
    expect(res.ok).toBe(true);
    const warn = toastStore.toasts.find((t) => t.kind === 'warn');
    expect(warn?.text).toBe(
      '2 keyboard shortcuts were not copied: review.queue.discard (d is the hotkey of class "dog"); cluster.undo (z is the hotkey of class "zebra").',
    );
  });

  it('shows no toast when no keymap action was dropped (absent or empty)', async () => {
    serve((url, init) =>
      init.method === 'POST'
        ? json({ project: ALPHA, keymap_clone_conflicts: [] })
        : undefined,
    );
    await createProjectsAdmin().cloneSettings(ALPHA, 'default', ['keymap']);
    expect(toastStore.toasts).toEqual([]);
  });

  it.each([
    ['slug_taken', 409, "a project named 'alpha' already exists"],
    ['slug_retired', 409, "'gone' was used by a deleted project"],
    ['shard_budget_exceeded', 409, 'OpenSearch has a 2 GB heap: no room'],
    ['slug_invalid', 422, "'Bad Slug' is not a valid project slug"],
  ])('returns a served %s refusal verbatim', async (code, status, message) => {
    serve((url, init) =>
      init.method === 'POST'
        ? json({ detail: { error: code, message } }, status)
        : undefined,
    );
    const res = await createProjectsAdmin().create({ slug: 'x', display_name: 'X' });
    expect(res).toEqual({ ok: false, code, message });
  });
});

describe('edit / archive / unarchive / clone', () => {
  it('edit sends only the changed fields plus the served revision', async () => {
    serve((url, init) =>
      init.method === 'PATCH'
        ? json({ project: { ...ALPHA, display_name: 'A2' } })
        : undefined,
    );
    await createProjectsAdmin().edit(ALPHA, { display_name: 'A2' });
    expect(writes()).toEqual([
      {
        method: 'PATCH',
        url: `${API_PREFIX}/projects/alpha`,
        body: { display_name: 'A2', expected_revision: 7 },
      },
    ]);
  });

  it('a 409 revision_conflict comes back as that code with the served message', async () => {
    serve((url, init) =>
      init.method === 'PATCH'
        ? json(
            {
              detail: {
                error: 'revision_conflict',
                message: 'expected revision 7, current is 9',
                current_revision: 9,
              },
            },
            409,
          )
        : undefined,
    );
    const res = await createProjectsAdmin().edit(ALPHA, { description: 'x' });
    expect(res).toEqual({
      ok: false,
      code: 'revision_conflict',
      message: 'expected revision 7, current is 9',
    });
  });

  it('archive / unarchive send the served revision; project_busy is verbatim', async () => {
    serve((url) =>
      url.endsWith('/archive')
        ? json(
            {
              detail: {
                error: 'project_busy',
                message: "'alpha' has 1 running job(s)",
                jobs: ['j1'],
              },
            },
            409,
          )
        : url.endsWith('/unarchive')
          ? json({ project: { ...ALPHA, status: 'active' } })
          : undefined,
    );
    const admin = createProjectsAdmin();
    expect(await admin.archive(ALPHA)).toEqual({
      ok: false,
      code: 'project_busy',
      message: "'alpha' has 1 running job(s)",
    });
    expect((await admin.unarchive(ALPHA)).ok).toBe(true);
    expect(writes().map((c) => [c.url, c.body])).toEqual([
      [`${API_PREFIX}/projects/alpha/archive`, { expected_revision: 7 }],
      [`${API_PREFIX}/projects/alpha/unarchive`, { expected_revision: 7 }],
    ]);
  });

  it('clone_settings posts {from, axes, expected_revision}; target_not_empty is verbatim', async () => {
    serve((url) =>
      url.endsWith('/clone_settings')
        ? json(
            {
              detail: {
                error: 'target_not_empty',
                message: 'classes only copy into an empty project',
              },
            },
            409,
          )
        : undefined,
    );
    const res = await createProjectsAdmin().cloneSettings(ALPHA, 'default', ['classes']);
    expect(res).toMatchObject({ ok: false, code: 'target_not_empty' });
    expect(writes()[0]!.body).toEqual({
      from: 'default',
      axes: ['classes'],
      expected_revision: 7,
    });
  });
});

describe('delete', () => {
  it('dry run returns the served report', async () => {
    const report = {
      indexes: [{ name: 'op_alpha_items', docs: 20 }],
      dirs: [],
      promoted_models: [],
      mlflow_experiment: 'alpha',
      running_jobs: [],
      referenced_by: [],
      blocking: ['last_active_project'],
      blocking_detail: [
        { code: 'last_active_project', message: 'this is the only remaining project' },
      ],
    };
    serve((url) => (url.includes('dry_run=true') ? json(report) : undefined));
    const res = await createProjectsAdmin().dryRunDelete(ALPHA);
    expect(res).toEqual({ ok: true, report });
  });

  it('a 409 project_protected shows the served message', async () => {
    serve((url) =>
      url.includes('confirm=')
        ? json(
            {
              detail: {
                error: 'project_protected',
                message: 'The default project can be archived but not deleted.',
                project: 'default',
              },
            },
            409,
          )
        : undefined,
    );
    const res = await createProjectsAdmin().remove(DEFAULT, 'default');
    expect(res).toEqual({
      ok: false,
      code: 'project_protected',
      message: 'The default project can be archived but not deleted.',
    });
    expect(writes()[0]!.url).toBe(`${API_PREFIX}/projects/default?confirm=default`);
  });

  it('a real delete sends what the operator typed as confirm', async () => {
    serve((url) =>
      url.includes('confirm=')
        ? json({ project: { ...ALPHA, status: 'deleting' } }, 202)
        : undefined,
    );
    const res = await createProjectsAdmin().remove(ALPHA, 'alpha');
    expect(res).toMatchObject({ ok: true, project: { status: 'deleting' } });
    expect(writes()).toEqual([
      {
        method: 'DELETE',
        url: `${API_PREFIX}/projects/alpha?confirm=alpha`,
        body: undefined,
      },
    ]);
  });
});

describe('polling while a row is transient', () => {
  afterEach(() => vi.useRealTimers());

  function serveSequence(statuses: string[]) {
    let n = 0;
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        const status = statuses[Math.min(n++, statuses.length - 1)]!;
        const rows =
          status === 'gone'
            ? [DEFAULT]
            : [DEFAULT, testProject({ slug: 'alpha', status: status as 'deleting' })];
        return json(testProjectsResponse(rows));
      }),
    );
    return () => n;
  }

  it('re-reads every 2 s while a row is deleting, then stops once it is gone', async () => {
    vi.useFakeTimers();
    const reads = serveSequence(['deleting', 'deleting', 'gone']);
    const admin = createProjectsAdmin();
    admin.start();
    await admin.load();
    expect(reads()).toBe(1);
    await vi.advanceTimersByTimeAsync(1999);
    expect(reads()).toBe(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(reads()).toBe(2);
    await vi.advanceTimersByTimeAsync(2000);
    expect(reads()).toBe(3);
    expect(admin.list.map((p) => p.slug)).toEqual(['default']);
    await vi.advanceTimersByTimeAsync(10000);
    expect(reads()).toBe(3);
    admin.stop();
  });

  it('also polls a building row', async () => {
    vi.useFakeTimers();
    const reads = serveSequence(['building', 'active']);
    const admin = createProjectsAdmin();
    admin.start();
    await admin.load();
    await vi.advanceTimersByTimeAsync(2000);
    expect(reads()).toBe(2);
    await vi.advanceTimersByTimeAsync(10000);
    expect(reads()).toBe(2);
    admin.stop();
  });

  it('stop() cancels a pending poll, and never polls before start()', async () => {
    vi.useFakeTimers();
    const reads = serveSequence(['deleting']);
    const idle = createProjectsAdmin();
    await idle.load();
    await vi.advanceTimersByTimeAsync(10000);
    expect(reads()).toBe(1);

    const admin = createProjectsAdmin();
    admin.start();
    await admin.load();
    admin.stop();
    await vi.advanceTimersByTimeAsync(10000);
    expect(reads()).toBe(2);
  });
});
