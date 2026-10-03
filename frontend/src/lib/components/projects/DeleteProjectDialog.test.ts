/**
 * Guarded delete dialog (v0.4.0): the dry run's served `referenced_by`
 * rows, a 409 `in_use` naming the served `projects` / `used_by`, and a 503
 * `config_store_unavailable` with its message and a Retry. The backend's
 * delete force is deliberately not offered.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import DeleteProjectDialog from './DeleteProjectDialog.svelte';
import { createProjectsAdmin } from '$lib/projects/projectsAdminController.svelte';
import { testProject } from '$lib/test/fixtures/projects';

const ALPHA = testProject({ slug: 'alpha', display_name: 'Alpha' });

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

const report = (over: Record<string, unknown> = {}) => ({
  indexes: [],
  dirs: [],
  promoted_models: [],
  mlflow_experiment: 'alpha',
  running_jobs: [],
  referenced_by: [],
  blocking: [],
  blocking_detail: [],
  ...over,
});

let mounted: ReturnType<typeof mount> | null = null;
afterEach(() => {
  if (mounted) unmount(mounted);
  mounted = null;
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
});

const q = <T extends Element = HTMLElement>(id: string) =>
  document.querySelector<T>(`[data-testid="${id}"]`);

function open(handler: (url: string) => Response) {
  const calls: string[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      calls.push(String(url));
      return handler(String(url));
    }),
  );
  mounted = mount(DeleteProjectDialog, {
    target: document.body,
    props: { project: ALPHA, admin: createProjectsAdmin(), onclose: vi.fn() },
  });
  flushSync();
  return calls;
}

function typeConfirmAndSubmit(): void {
  const input = q<HTMLInputElement>('delete-project-confirm')!;
  input.value = 'alpha';
  input.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
  q('delete-project-submit')!.closest('form')!.requestSubmit();
}

describe('DeleteProjectDialog', () => {
  it('lists the dry run referenced_by rows with served project and profile', async () => {
    open((url) =>
      url.includes('dry_run=true')
        ? json(
            report({
              referenced_by: [
                { project: 'beta', profile: 'tags_v2' },
                { project: 'gamma', profile: null },
              ],
            }),
          )
        : json({}),
    );
    await vi.waitFor(() => expect(q('delete-project-references')).not.toBeNull());
    const text = q('delete-project-references')!.textContent!;
    expect(text).toContain('beta');
    expect(text).toContain('tags_v2');
    expect(text).toContain('gamma');
    expect(q('delete-project-references')!.querySelectorAll('li')).toHaveLength(2);
  });

  it('renders no references block when none are served', async () => {
    open(() => json(report()));
    await vi.waitFor(() => expect(q('delete-project-confirm')).not.toBeNull());
    expect(q('delete-project-references')).toBeNull();
  });

  it('a 409 in_use shows the served message, projects and used_by', async () => {
    open((url) =>
      url.includes('dry_run=true')
        ? json(report())
        : json(
            {
              detail: {
                error: 'in_use',
                message: 'a model of this project is in use elsewhere',
                projects: ['beta'],
                used_by: [{ project: 'beta', profile: 'tags_v2' }],
              },
            },
            409,
          ),
    );
    await vi.waitFor(() => expect(q('delete-project-confirm')).not.toBeNull());
    typeConfirmAndSubmit();
    await vi.waitFor(() => expect(q('delete-project-in-use')).not.toBeNull());
    expect(q('delete-project-error')!.textContent).toBe(
      'a model of this project is in use elsewhere',
    );
    const text = q('delete-project-in-use')!.textContent!;
    expect(text).toContain('beta');
    expect(text).toContain('tags_v2');
  });

  it('a 503 config_store_unavailable shows its message and Retry, with no force', async () => {
    // apiFetch retries a 5xx itself, so the outage is a flag, not a call count.
    let down = true;
    const calls = open((url) => {
      if (url.includes('dry_run=true')) {
        return down
          ? json(
              {
                detail: {
                  error: 'config_store_unavailable',
                  message: 'could not read every project',
                },
              },
              503,
            )
          : json(report());
      }
      return json({}, 500);
    });
    await vi.waitFor(() => expect(q('delete-project-dry-run-error')).not.toBeNull(), {
      timeout: 5000,
    });
    expect(q('delete-project-dry-run-error')!.textContent).toBe(
      'could not read every project',
    );
    expect(document.body.textContent).not.toMatch(/anyway|force/i);
    const before = calls.length;
    down = false;
    q('delete-project-retry')!.click();
    flushSync();
    await vi.waitFor(() => expect(q('delete-project-confirm')).not.toBeNull());
    expect(calls.length).toBe(before + 1);
    expect(q('delete-project-retry')).toBeNull();
  });

  it('a 503 on the real delete offers Retry that resends the same confirm', async () => {
    let down = true;
    const calls = open((url) => {
      if (url.includes('dry_run=true')) return json(report());
      return down
        ? json(
            { detail: { error: 'config_store_unavailable', message: 'store down' } },
            503,
          )
        : json({ project: { ...ALPHA, status: 'deleting' } }, 202);
    });
    await vi.waitFor(() => expect(q('delete-project-confirm')).not.toBeNull());
    typeConfirmAndSubmit();
    await vi.waitFor(() => expect(q('delete-project-retry')).not.toBeNull(), {
      timeout: 5000,
    });
    expect(q('delete-project-error')!.textContent).toBe('store down');
    expect(document.body.textContent).not.toMatch(/anyway|force/i);
    const before = calls.filter((u) => u.includes('confirm=alpha')).length;
    down = false;
    q('delete-project-retry')!.click();
    await vi.waitFor(() =>
      expect(calls.filter((u) => u.includes('confirm=alpha'))).toHaveLength(before + 1),
    );
  });
});
