/**
 * `/models` cross-project sharing (projects P2, §5.5), mounted against a
 * stateful stub of `GET {scoped}/models/status?include_other_projects=true`,
 * `PUT .../sharing` and `GET .../class_mapping`:
 *
 * - the owner-only toggle renders only for the active project's own model
 *   AND only with a served sharing revision, and sends that revision;
 * - another project's shared model shows its project chip, the served
 *   mapped count and unmapped names;
 * - every refusal renders the served message verbatim; a
 *   `revision_conflict` reloads so the next attempt carries the fresh
 *   revision; `in_use` offers the served `force`;
 * - the class-mapping view shows names, never class ids.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ModelsPage from './+page.svelte';
import { projectsStore } from '$stores/projects.svelte';
import { testProject } from '$lib/test/fixtures/projects';
import type { ProjectSummary } from '$lib/types_projects';

const ACTIVE = projectsStore.current!;

const BASE = {
  role: 'Promoted',
  kind: 'triton',
  model_type: 'Promoted checkpoint',
  status: 'ready',
  version: '1',
  inference_count: 0,
  exec_count: 0,
  inference_failed: 0,
  avg_latency_ms: null,
  last_error: null,
  endpoint: 'http://triton:8000',
  is_region_protected: false,
  requires_force_to_unload: false,
  unloadable: true,
  optional: false,
  shared: false,
  project: null as string | null,
  owned: false,
  sharing_revision: null as number | null,
  class_mapping: null as unknown,
};

interface Row extends Record<string, unknown> {
  name: string;
}

let rows: Row[];
let writes: { method: string; url: string; body: unknown }[];
let statusReads: string[];
let putHandler: (
  url: string,
  body: { shared: boolean; expected_revision: number },
) => Response;
let savedList: ProjectSummary[];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function own(): Row {
  return rows.find((r) => r.name === 'own_det')!;
}

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;

function q<T extends Element = HTMLElement>(testId: string): T | null {
  return target.querySelector<T>(`[data-testid="${testId}"]`);
}

function click(testId: string): void {
  q<HTMLElement>(testId)!.click();
  flushSync();
}

async function render(): Promise<void> {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ModelsPage, { target });
  flushSync();
  await vi.waitFor(() => expect(q('model-sharing-own_det')).not.toBeNull());
}

beforeEach(() => {
  savedList = projectsStore.list;
  projectsStore.list = [ACTIVE, testProject({ slug: 'beta', display_name: 'Beta lot' })];
  rows = [
    {
      ...BASE,
      name: 'own_det',
      friendly_name: 'own_det (promoted)',
      project: ACTIVE.slug,
      owned: true,
      shared: false,
      sharing_revision: 4,
      class_mapping: { mapped_count: 3, unmapped: [] },
    },
    {
      ...BASE,
      name: 'own_legacy',
      friendly_name: 'own_legacy (promoted)',
      project: ACTIVE.slug,
      owned: true,
      shared: true,
      sharing_revision: null,
      class_mapping: null,
    },
    {
      // A legacy promote.json with no `project`: the server still says it
      // owns it, so the toggle shows (it used to hide for a null project).
      ...BASE,
      name: 'own_unstamped',
      friendly_name: 'own_unstamped (promoted)',
      project: null,
      owned: true,
      shared: false,
      sharing_revision: 1,
    },
    {
      ...BASE,
      name: 'beta__det',
      friendly_name: 'beta__det (shared by beta)',
      project: 'beta',
      owned: false,
      shared: true,
      unloadable: false,
      class_mapping: { mapped_count: 2, unmapped: ['van', 'bus'] },
    },
    { ...BASE, name: 'encoder', friendly_name: 'Encoder' },
  ];
  writes = [];
  statusReads = [];
  putHandler = (_url, body) => {
    const r = own();
    r.shared = body.shared;
    r.sharing_revision = (r.sharing_revision as number) + 1;
    return json({
      name: 'own_det',
      project: ACTIVE.slug,
      shared: body.shared,
      revision: r.sharing_revision,
      used_by: [],
    });
  };
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const u = String(url);
      const method = init.method ?? 'GET';
      // The page's one-shot VLM registry probe: absent on this backend.
      if (method === 'GET' && u.endsWith('/vlm/endpoints')) {
        return json({ detail: 'Not Found' }, 404);
      }
      if (method === 'GET' && u.includes('/models/status')) {
        statusReads.push(u);
        return json({ models: rows.map((r) => ({ ...r })) });
      }
      if (method === 'GET' && u.endsWith('/models/beta__det/class_mapping')) {
        return json({
          model: 'beta__det',
          model_project: 'beta',
          project: ACTIVE.slug,
          entries: [
            {
              model_id: 0,
              model_name: 'Crate',
              class_id: 71,
              class_name: 'crate',
              match: 'case_insensitive',
            },
            {
              model_id: 1,
              model_name: 'van',
              class_id: null,
              class_name: null,
              match: 'none',
            },
          ],
          unmapped: ['van'],
          not_covered: ['pallet'],
          labels: {
            match: {
              exact: 'Same name',
              case_insensitive: 'Same name, different case',
              none: 'No match',
            },
          },
        });
      }
      const body = init.body ? JSON.parse(String(init.body)) : undefined;
      writes.push({ method, url: u, body });
      if (method === 'PUT' && u.includes('/sharing')) return putHandler(u, body);
      return json({ detail: 'unrouted' }, 500);
    }),
  );
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
  vi.unstubAllGlobals();
  projectsStore.list = savedList;
});

describe('/models sharing', () => {
  it("lists other projects' shared models and shows served sharing state", async () => {
    await render();
    expect(statusReads[0]).toContain('include_other_projects=true');

    // Own model with a served revision: state + toggle.
    const mine = q('model-sharing-own_det')!;
    expect(mine.querySelector('[data-testid="model-sharing-state"]')!.textContent).toBe(
      'Not shared',
    );
    expect(q('model-share-toggle-own_det')!.textContent).toBe(
      'Share with other projects',
    );
    expect(mine.querySelector('[data-testid="model-mapped-count"]')!.textContent).toBe(
      '3 classes map',
    );
    expect(mine.querySelector('[data-testid="model-unmapped"]')).toBeNull();

    // Owned by the server's own verdict even with no served `project`.
    expect(q('model-share-toggle-own_unstamped')!.textContent).toBe(
      'Share with other projects',
    );

    // Own model WITHOUT a served revision: state, but no toggle.
    expect(q('model-sharing-own_legacy')!.textContent).toContain(
      'Shared with other projects',
    );
    expect(q('model-share-toggle-own_legacy')).toBeNull();

    // Another project's shared model: project chip, never a toggle.
    const theirs = q('model-sharing-beta__det')!;
    expect(theirs.querySelector('[data-testid="model-project-chip"]')!.textContent).toBe(
      'from Beta lot',
    );
    expect(q('model-share-toggle-beta__det')).toBeNull();
    expect(theirs.querySelector('[data-testid="model-mapped-count"]')!.textContent).toBe(
      '2 classes map',
    );
    expect(
      theirs.querySelector('[data-testid="model-unmapped-names"]')!.textContent!.trim(),
    ).toBe('van, bus');

    // No project, no class list: no sharing section at all.
    expect(q('model-sharing-encoder')).toBeNull();
  });

  it("another project's model offers no Unload: the server serves unloadable: false for it", async () => {
    await render();
    const unloadButtons = (testId: string) =>
      [...q(testId)!.closest('li')!.querySelectorAll('button')].filter(
        (b) => b.textContent?.trim() === 'Unload',
      );
    // The foreign model's card has no Unload; the owned one does.
    expect(unloadButtons('model-sharing-beta__det')).toHaveLength(0);
    expect(unloadButtons('model-sharing-own_det').length).toBeGreaterThan(0);
  });

  it('share is confirm-gated and sends the served revision, then reloads', async () => {
    await render();
    click('model-share-toggle-own_det');
    expect(q('share-model-dialog')).not.toBeNull();
    expect(writes).toEqual([]);
    const reads = statusReads.length;
    click('share-model-confirm');
    await vi.waitFor(() => expect(q('share-model-dialog')).toBeNull());
    expect(writes).toEqual([
      {
        method: 'PUT',
        url: expect.stringMatching(/\/models\/own_det\/sharing$/),
        body: { shared: true, expected_revision: 4 },
      },
    ]);
    expect(statusReads.length).toBe(reads + 1);
    await vi.waitFor(() =>
      expect(
        q('model-sharing-own_det')!.querySelector('[data-testid="model-sharing-state"]')!
          .textContent,
      ).toBe('Shared with other projects'),
    );
  });

  it('unsharing says the server names any project using the model', async () => {
    own().shared = true;
    await render();
    click('model-share-toggle-own_det');
    const text = q('share-model-text')!.textContent!;
    expect(text).toMatch(/active detection profile/i);
    expect(text).not.toMatch(/may be using it/i);
    expect(text).not.toMatch(/\bsafe/i);
  });

  it('a revision_conflict shows the served message, reloads, and retries with the fresh revision', async () => {
    let first = true;
    const accept = putHandler;
    putHandler = (url, body) => {
      if (first) {
        first = false;
        own().sharing_revision = 9;
        return json(
          {
            detail: {
              error: 'revision_conflict',
              message: "'own_det' sharing revision is 9, not 4",
              current_revision: 9,
            },
          },
          409,
        );
      }
      return accept(url, body);
    };
    await render();
    click('model-share-toggle-own_det');
    click('share-model-confirm');
    await vi.waitFor(() =>
      expect(q('share-model-error')?.textContent).toContain(
        "'own_det' sharing revision is 9, not 4",
      ),
    );
    expect(q('share-model-reloaded')).not.toBeNull();
    // The reload landed: the dialog now reads the fresh served entry.
    await vi.waitFor(() => expect(statusReads.length).toBeGreaterThanOrEqual(2));
    click('share-model-confirm');
    await vi.waitFor(() => expect(q('share-model-dialog')).toBeNull());
    expect(writes.map((w) => w.body)).toEqual([
      { shared: true, expected_revision: 4 },
      { shared: true, expected_revision: 9 },
    ]);
  });

  it('a served refusal renders verbatim', async () => {
    putHandler = () =>
      json(
        {
          detail: {
            error: 'model_not_found',
            message: "'own_det' is not a model owned by this project",
          },
        },
        404,
      );
    await render();
    click('model-share-toggle-own_det');
    click('share-model-confirm');
    await vi.waitFor(() =>
      expect(q('share-model-error')?.textContent?.trim()).toBe(
        "'own_det' is not a model owned by this project",
      ),
    );
    expect(q('share-model-force')).toBeNull();
  });

  it('in_use lists the served projects and offers the served force', async () => {
    own().shared = true;
    const accept = putHandler;
    putHandler = (url, body) =>
      url.includes('force=true')
        ? accept(url, body)
        : json(
            {
              detail: {
                error: 'in_use',
                message: "'own_det' is still used by 1 other project(s)",
                projects: ['beta'],
              },
            },
            409,
          );
    await render();
    click('model-share-toggle-own_det');
    click('share-model-confirm');
    await vi.waitFor(() =>
      expect(q('share-model-in-use')?.textContent).toContain('beta'),
    );
    click('share-model-force');
    // Arming only reveals the warning; nothing is sent until the confirm.
    expect(q('share-model-force-warning')?.textContent).toContain('beta');
    expect(writes).toHaveLength(1);
    click('share-model-force-confirm');
    await vi.waitFor(() => expect(q('share-model-dialog')).toBeNull());
    expect(writes.map((w) => w.url.split('/').slice(-1)[0])).toEqual([
      'sharing',
      'sharing?force=true',
    ]);
    expect(writes[1]!.body).toEqual({ shared: false, expected_revision: 4 });
  });

  it('an unreadable project (503 config_store_unavailable) shows the served message and no force', async () => {
    own().shared = true;
    putHandler = () =>
      json(
        {
          detail: {
            error: 'config_store_unavailable',
            message:
              'could not read every project to see who uses this model; retry, or pass force',
          },
        },
        503,
      );
    await render();
    click('model-share-toggle-own_det');
    click('share-model-confirm');
    // apiFetch retries a 5xx with backoff before surfacing it.
    await vi.waitFor(
      () => expect(q('share-model-error')?.textContent).toContain('retry, or pass force'),
      { timeout: 5000 },
    );
    expect(q('share-model-force')).toBeNull();
  });

  it('the class mapping view shows names and served match labels, never ids', async () => {
    await render();
    const details = q('model-sharing-beta__det')!.querySelector<HTMLDetailsElement>(
      '[data-testid="model-mapping-details"]',
    )!;
    details.open = true;
    details.dispatchEvent(new Event('toggle'));
    await vi.waitFor(() =>
      expect(details.querySelector('[data-testid="model-mapping-table"]')).not.toBeNull(),
    );
    const text = details.textContent!;
    expect(text).toContain('Crate');
    expect(text).toContain('crate');
    expect(text).toContain('Same name, different case');
    expect(text).toContain('No match');
    expect(
      details.querySelector('[data-testid="model-not-covered"]')!.textContent,
    ).toContain('pallet');
    expect(text).not.toContain('71');
  });
});
