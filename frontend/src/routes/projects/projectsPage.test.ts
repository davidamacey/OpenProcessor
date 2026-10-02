/**
 * `/projects` mounted against a stubbed `fetch`: every row action is
 * gated on served flags only, a served `blocked` capacity disables
 * Create, and each dialog renders the server's own refusal — including
 * `revision_conflict` → reload → resubmit with the fresh revision, and
 * the delete dry run's `blocking` reasons.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import Page from './+page.svelte';
import { API_PREFIX } from '$lib/api';
import { combineAvailability } from '$lib/combine/combineAvailability.svelte';
import { projectPauseStore } from '$stores/projectPause.svelte';
import { projectsStore } from '$stores/projects.svelte';
import { toastStore } from '$stores/toast.svelte';
import {
  TEST_LIMITS,
  testCapacity,
  testProject,
  testProjectsResponse,
} from '$lib/test/fixtures/projects';
import type { ProjectSummary } from '$lib/types_projects';

const DEFAULT = testProject({
  slug: 'default',
  prefix: API_PREFIX,
  is_default: true,
  deletable: false,
  display_name: 'Default',
});
const ALPHA = testProject({ slug: 'alpha', display_name: 'Alpha', revision: 4 });
const OLD = testProject({ slug: 'old', status: 'archived', writable: false });
const WIP = testProject({
  slug: 'wip',
  status: 'building',
  writable: false,
  selectable: false,
});

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

type Handler = (url: string, init: RequestInit) => Response | undefined;
let listed: ProjectSummary[];
let capacity = testCapacity('ok');
let handler: Handler = () => undefined;
let writes: { method: string; url: string; body: unknown }[] = [];
/** The served per-project pause flag, carried on the `GET /projects` rows. */
let paused: Record<string, boolean> = {};
/** What `GET /projects/combine/<sentinel>` answers (the P4 gate probe). */
let combineMounted = false;
let combineProbes = 0;
/** Served `limits.cloneable_axes` override (null = the fixture's own). */
let cloneAxes: string[] | null = null;

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;

function q<T extends Element = HTMLElement>(testId: string): T | null {
  return document.querySelector<T>(`[data-testid="${testId}"]`);
}

async function render(): Promise<void> {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(Page, { target });
  flushSync();
  await vi.waitFor(() => expect(q('project-row-default')).not.toBeNull());
}

function click(testId: string): void {
  q<HTMLElement>(testId)!.click();
  flushSync();
}

function type(testId: string, value: string): void {
  const el = q<HTMLInputElement>(testId)!;
  el.value = value;
  el.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
}

function submit(testId: string): void {
  q<HTMLButtonElement>(testId)!.closest('form')!.requestSubmit();
  flushSync();
}

beforeEach(() => {
  listed = [DEFAULT, ALPHA, OLD, WIP];
  capacity = testCapacity('ok');
  handler = () => undefined;
  writes = [];
  paused = {};
  combineMounted = false;
  combineProbes = 0;
  cloneAxes = null;
  combineAvailability.reset();
  projectPauseStore.reset();
  toastStore.toasts = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const method = init.method ?? 'GET';
      const u = String(url);
      if (method === 'GET' && /\/projects(\?|$)/.test(u)) {
        return json(
          testProjectsResponse(
            listed.map((p) => ({ ...p, paused: paused[p.slug] ?? false })),
            {
              capacity,
              ...(cloneAxes
                ? { limits: { ...TEST_LIMITS, cloneable_axes: cloneAxes } }
                : {}),
            },
          ),
        );
      }
      if (method === 'GET' && /\/projects\/combine\/[^/?]+$/.test(u)) {
        combineProbes += 1;
        return combineMounted
          ? json(
              { detail: { error: 'combine_not_found', message: 'no combine job' } },
              404,
            )
          : json({ detail: 'Not Found' }, 404);
      }
      writes.push({
        method,
        url: u,
        body: init.body ? JSON.parse(String(init.body)) : undefined,
      });
      return handler(u, init) ?? json({ detail: 'unrouted' }, 500);
    }),
  );
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
  vi.unstubAllGlobals();
  projectsStore.select(DEFAULT);
});

describe('row actions follow served flags only', () => {
  it('offers exactly what each served project allows', async () => {
    await render();
    // default: selectable + writable, but deletable: false.
    expect(q('project-open-default')).not.toBeNull();
    expect(q('project-archive-default')).not.toBeNull();
    expect(q('project-delete-default')).toBeNull();
    // alpha: active and deletable.
    expect(q('project-delete-alpha')).not.toBeNull();
    expect(q('project-clone-alpha')).not.toBeNull();
    // archived: selectable but not writable → Unarchive, no Archive/Copy.
    expect(q('project-unarchive-old')).not.toBeNull();
    expect(q('project-archive-old')).toBeNull();
    expect(q('project-clone-old')).toBeNull();
    // building: not selectable → nothing to open, edit or archive.
    expect(q('project-open-wip')).toBeNull();
    expect(q('project-edit-wip')).toBeNull();
    expect(q('project-unarchive-wip')).toBeNull();
    expect(q('project-status-wip')!.textContent).toBe('Building');
    expect(q('project-open-alpha')!.getAttribute('href')).toBe('/p/alpha/dashboard');
  });

  it('Archive / Unarchive read the served archivable / unarchivable, not status', async () => {
    // Flags that contradict what status/writable would suggest: the served
    // flags win, so no client-side inference can survive this test.
    listed = [
      DEFAULT,
      testProject({ slug: 'pinned', status: 'active', archivable: false }),
      testProject({
        slug: 'frozen',
        status: 'archived',
        writable: false,
        unarchivable: false,
      }),
      testProject({
        slug: 'odd',
        status: 'building',
        writable: false,
        selectable: false,
        unarchivable: true,
      }),
    ];
    await render();
    expect(q('project-archive-pinned')).toBeNull();
    expect(q('project-clone-pinned')).not.toBeNull();
    expect(q('project-unarchive-frozen')).toBeNull();
    expect(q('project-unarchive-odd')).not.toBeNull();
  });

  it('a served blocked capacity disables Create and shows the served message', async () => {
    capacity = testCapacity('blocked');
    await render();
    expect(q<HTMLButtonElement>('projects-create')!.disabled).toBe(true);
    expect(q('projects-capacity')!.textContent).toContain(
      'capacity is blocked: served message',
    );
    expect(q('projects-capacity')!.textContent).toContain('No room for another project');
  });
});

describe('create', () => {
  it('renders a served refusal verbatim in the dialog', async () => {
    handler = () =>
      json(
        {
          detail: {
            error: 'slug_taken',
            message: "a project named 'alpha' already exists",
          },
        },
        409,
      );
    await render();
    click('projects-create');
    type('create-project-slug', 'alpha');
    type('create-project-name', 'Alpha again');
    submit('create-project-submit');
    await vi.waitFor(() =>
      expect(q('create-project-error')?.textContent).toBe(
        "a project named 'alpha' already exists",
      ),
    );
    expect(writes[0]).toMatchObject({
      method: 'POST',
      body: { slug: 'alpha', display_name: 'Alpha again', description: '' },
    });
  });

  it('flags a served reserved slug before submit, without blocking it', async () => {
    await render();
    click('projects-create');
    type('create-project-slug', 'projects');
    expect(q('create-project-slug-hint')!.textContent).toContain('reserved');
  });

  it('warn capacity: shows the served message, creates, and toasts the 201 warnings', async () => {
    capacity = testCapacity('warn');
    handler = () =>
      json(
        {
          project: testProject({ slug: 'beta', display_name: 'Beta' }),
          warnings: [{ code: 'shard_budget_high', message: 'served: near the budget' }],
        },
        201,
      );
    await render();
    click('projects-create');
    expect(q('create-project-capacity')!.textContent).toContain(
      'capacity is warn: served message',
    );
    type('create-project-slug', 'beta');
    type('create-project-name', 'Beta');
    submit('create-project-submit');
    await vi.waitFor(() => expect(q('create-project-dialog')).toBeNull());
    expect(toastStore.toasts.map((t) => t.text)).toContain('served: near the budget');
  });
});

describe('edit', () => {
  it('a revision_conflict offers a reload, after which the save carries the fresh revision', async () => {
    let conflictOnce = true;
    handler = (url, init) => {
      if (init.method !== 'PATCH') return undefined;
      if (conflictOnce) {
        conflictOnce = false;
        listed = [DEFAULT, { ...ALPHA, revision: 9 }, OLD, WIP];
        return json(
          {
            detail: {
              error: 'revision_conflict',
              message: 'expected revision 4, current is 9',
            },
          },
          409,
        );
      }
      return json({ project: { ...ALPHA, revision: 10, display_name: 'Renamed' } });
    };
    await render();
    click('project-edit-alpha');
    type('edit-project-name', 'Renamed');
    submit('edit-project-submit');
    await vi.waitFor(() =>
      expect(q('edit-project-error')?.textContent).toContain(
        'expected revision 4, current is 9',
      ),
    );
    expect(q<HTMLButtonElement>('edit-project-submit')!.disabled).toBe(true);
    click('edit-project-reload');
    await vi.waitFor(() => expect(q('edit-project-reload')).toBeNull());
    // The operator's edit survives the reload.
    expect(q<HTMLInputElement>('edit-project-name')!.value).toBe('Renamed');
    submit('edit-project-submit');
    await vi.waitFor(() => expect(q('edit-project-dialog')).toBeNull());
    expect(writes.map((w) => w.body)).toEqual([
      { display_name: 'Renamed', expected_revision: 4 },
      { display_name: 'Renamed', expected_revision: 9 },
    ]);
  });
});

describe('delete', () => {
  it('shows the served blocking reasons and no confirm field while anything blocks', async () => {
    handler = (url) =>
      url.includes('dry_run=true')
        ? json({
            indexes: [{ name: 'op_alpha_items', docs: 20 }],
            dirs: [],
            promoted_models: [],
            mlflow_experiment: 'alpha',
            running_jobs: [{ id: 'j1' }],
            referenced_by: [],
            blocking: ['project_busy'],
            blocking_detail: [
              { code: 'project_busy', message: '1 job(s) still running' },
            ],
          })
        : undefined;
    await render();
    click('project-delete-alpha');
    await vi.waitFor(() =>
      expect(q('delete-project-blocking')?.textContent).toContain(
        '1 job(s) still running',
      ),
    );
    expect(q('delete-project-confirm')).toBeNull();
    expect(q<HTMLButtonElement>('delete-project-submit')!.disabled).toBe(true);
    expect(q('delete-project-report')!.textContent).toContain('op_alpha_items');
  });

  it('sends the typed slug as confirm and renders a served refusal verbatim', async () => {
    handler = (url) => {
      if (url.includes('dry_run=true')) {
        return json({
          indexes: [],
          dirs: [],
          promoted_models: [],
          mlflow_experiment: 'alpha',
          running_jobs: [],
          referenced_by: [],
          blocking: [],
          blocking_detail: [],
        });
      }
      return json(
        {
          detail: {
            error: 'project_protected',
            message: 'This project can be archived but not deleted.',
          },
        },
        409,
      );
    };
    await render();
    click('project-delete-alpha');
    await vi.waitFor(() => expect(q('delete-project-confirm')).not.toBeNull());
    type('delete-project-confirm', 'alpha');
    submit('delete-project-submit');
    await vi.waitFor(() =>
      expect(q('delete-project-error')?.textContent).toBe(
        'This project can be archived but not deleted.',
      ),
    );
    expect(writes.map((w) => [w.method, w.url])).toEqual([
      ['DELETE', `${API_PREFIX}/projects/alpha?dry_run=true`],
      ['DELETE', `${API_PREFIX}/projects/alpha?confirm=alpha`],
    ]);
  });
});

describe('pipeline pause', () => {
  it('shows the served per-row flag with no per-row pause reads', async () => {
    paused = { alpha: true };
    await render();
    await vi.waitFor(() => expect(q('project-paused-alpha')).not.toBeNull());
    expect(q('project-paused-default')).toBeNull();
    // Writable rows offer the opposite of the served flag; the archived
    // row shows no control.
    expect(q('project-resume-alpha')).not.toBeNull();
    expect(q('project-pause-alpha')).toBeNull();
    expect(q('project-pause-default')).not.toBeNull();
    expect(q('project-pause-old')).toBeNull();
    expect(q('project-resume-old')).toBeNull();
    expect(writes).toEqual([]);
  });

  it("pause is confirm-gated, POSTs the row's served prefix, and shows the chip", async () => {
    handler = (u) =>
      u === `${ALPHA.prefix}/pause`
        ? ((paused.alpha = true),
          json({ project: 'alpha', paused: true, paused_by: ['project'], reason: null }))
        : undefined;
    await render();
    await vi.waitFor(() => expect(q('project-pause-alpha')).not.toBeNull());
    click('project-pause-alpha');
    expect(q('pause-project-dialog')).not.toBeNull();
    expect(writes).toEqual([]);
    click('pause-project-confirm');
    await vi.waitFor(() => expect(q('project-paused-alpha')).not.toBeNull());
    expect(writes).toEqual([
      { method: 'POST', url: `${ALPHA.prefix}/pause`, body: undefined },
    ]);
    expect(q('pause-project-dialog')).toBeNull();
    expect(q('project-resume-alpha')).not.toBeNull();
  });

  it('resume POSTs /resume and clears the chip from the served answer', async () => {
    paused = { alpha: true };
    handler = (u) =>
      u === `${ALPHA.prefix}/resume`
        ? ((paused.alpha = false),
          json({ project: 'alpha', paused: false, paused_by: [], reason: null }))
        : undefined;
    await render();
    await vi.waitFor(() => expect(q('project-resume-alpha')).not.toBeNull());
    click('project-resume-alpha');
    click('pause-project-confirm');
    await vi.waitFor(() => expect(q('project-paused-alpha')).toBeNull());
    expect(writes).toEqual([
      { method: 'POST', url: `${ALPHA.prefix}/resume`, body: undefined },
    ]);
  });

  it('a served refusal renders verbatim and the flag is unchanged', async () => {
    handler = () =>
      json(
        {
          detail: { error: 'project_read_only', message: 'served: project is read-only' },
        },
        409,
      );
    await render();
    await vi.waitFor(() => expect(q('project-pause-alpha')).not.toBeNull());
    click('project-pause-alpha');
    click('pause-project-confirm');
    await vi.waitFor(() =>
      expect(q('pause-project-error')?.textContent?.trim()).toBe(
        'served: project is read-only',
      ),
    );
    expect(q('project-paused-alpha')).toBeNull();
  });
});

describe('combine entry points (P4)', () => {
  it('offers Combine projects only when the combine router answers the probe', async () => {
    await render();
    await vi.waitFor(() => expect(combineProbes).toBe(1));
    await vi.waitFor(() => expect(combineAvailability.available).toBe(false));
    flushSync();
    expect(q('projects-combine')).toBeNull();
    unmount(instance!);
    instance = null;
    target.remove();

    combineMounted = true;
    combineAvailability.reset();
    await render();
    await vi.waitFor(() => expect(q('projects-combine')).not.toBeNull());
    expect(q('projects-combine')!.getAttribute('href')).toBe('/projects/combine');
  });

  it('links a project made by a combine to its job and labels its delete "Undo combine"', async () => {
    listed = [
      DEFAULT,
      testProject({
        slug: 'merged',
        display_name: 'Merged',
        origin: {
          kind: 'combine',
          job_id: 'cmb_20261001T120000_1a2b3c4d',
          sources: ['alpha'],
        },
      }),
      testProject({ slug: 'beta', origin: { kind: 'import', job_id: 'imp_1' } }),
      ALPHA,
    ];
    handler = (url) =>
      url.includes('dry_run=true')
        ? json({
            indexes: [],
            dirs: [],
            promoted_models: [],
            mlflow_experiment: null,
            running_jobs: [],
            referenced_by: [],
            blocking: [],
            blocking_detail: [],
          })
        : undefined;
    await render();
    expect(q('project-combine-job-merged')!.getAttribute('href')).toBe(
      '/projects/combine/cmb_20261001T120000_1a2b3c4d',
    );
    expect(q('project-combine-job-beta')).toBeNull();
    expect(q('project-combine-job-alpha')).toBeNull();
    expect(q('project-delete-merged')!.textContent?.trim()).toBe('Undo combine');
    expect(q('project-delete-beta')!.textContent?.trim()).toBe('Delete');
    expect(q('project-delete-alpha')!.textContent?.trim()).toBe('Delete');

    click('project-delete-merged');
    await vi.waitFor(() => expect(q('delete-project-title')).not.toBeNull());
    expect(q('delete-project-title')!.textContent).toContain('Undo combine');
    expect(q('delete-project-dialog')!.getAttribute('aria-label')).toBe('Undo combine');
    // The guarded flow is unchanged: a served dry run, then the typed slug.
    await vi.waitFor(() => expect(q('delete-project-confirm')).not.toBeNull());
  });

  it('a project with a combine origin but no job id gets no link', async () => {
    listed = [DEFAULT, testProject({ slug: 'orphan', origin: { kind: 'combine' } })];
    await render();
    expect(q('project-combine-job-orphan')).toBeNull();
    expect(q('project-delete-orphan')!.textContent?.trim()).toBe('Delete');
  });
});

describe('copy settings with the served vlm_activation axis (W9)', () => {
  it('offers the served axis and renders a 422 vlm_external_not_acknowledged verbatim', async () => {
    cloneAxes = ['settings_defaults', 'vlm_activation'];
    handler = (url, init) =>
      init.method === 'POST' && url.endsWith('/projects/alpha/clone_settings')
        ? json(
            {
              detail: {
                error: 'vlm_external_not_acknowledged',
                message:
                  "served: 'cloud_vlm' sends crops outside this deployment; acknowledge it first",
                endpoint: 'cloud_vlm',
                activate_via: '/vlm/endpoints/cloud_vlm/activate',
              },
            },
            422,
          )
        : undefined;
    await render();
    click('project-clone-alpha');
    expect(q('clone-settings-axis-vlm_activation')).not.toBeNull();
    submit('clone-settings-submit');
    await vi.waitFor(() =>
      expect(q('clone-settings-error')?.textContent).toBe(
        "served: 'cloud_vlm' sends crops outside this deployment; acknowledge it first",
      ),
    );
    expect(writes[0]!.body).toEqual({
      from: 'default',
      axes: ['settings_defaults', 'vlm_activation'],
      expected_revision: 4,
    });
  });
});
