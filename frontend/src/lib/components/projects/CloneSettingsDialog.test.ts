/**
 * Copy-settings dialog: no source is preselected (copying from a real
 * project must be an explicit pick), and a list reload — including the one
 * a submit itself triggers — never resets the operator's choices.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import CloneSettingsDialog from './CloneSettingsDialog.svelte';
import { createProjectsAdmin } from '$lib/projects/projectsAdminController.svelte';
import { testProject, testProjectsResponse } from '$lib/test/fixtures/projects';

const DEFAULT = testProject({ slug: 'default', is_default: true });
const ALPHA = testProject({ slug: 'alpha', display_name: 'Alpha' });
const BETA = testProject({ slug: 'beta', display_name: 'Beta' });

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

let mounted: ReturnType<typeof mount> | null = null;
afterEach(() => {
  if (mounted) unmount(mounted);
  mounted = null;
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
});

const q = <T extends Element = HTMLElement>(id: string) =>
  document.querySelector<T>(`[data-testid="${id}"]`);

async function open(rows = [DEFAULT, ALPHA, BETA], clonePost?: () => Promise<Response>) {
  let listRows = rows;
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      if ((init.method ?? 'GET') === 'POST' && String(url).includes('clone_settings')) {
        return clonePost ? clonePost() : json({ project: ALPHA, warnings: [] });
      }
      return json(testProjectsResponse(listRows));
    }),
  );
  const admin = createProjectsAdmin();
  await admin.load();
  const onclose = vi.fn();
  mounted = mount(CloneSettingsDialog, {
    target: document.body,
    props: { project: ALPHA, admin, onclose },
  });
  flushSync();
  return { admin, onclose, setRows: (r: typeof rows) => (listRows = r) };
}

describe('CloneSettingsDialog', () => {
  it('preselects no source and keeps Copy disabled until one is picked', async () => {
    await open();
    const select = q<HTMLSelectElement>('clone-settings-from')!;
    expect(select.value).toBe('');
    expect(q<HTMLButtonElement>('clone-settings-submit')!.disabled).toBe(true);

    select.value = 'default';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(q<HTMLButtonElement>('clone-settings-submit')!.disabled).toBe(false);
  });

  it('keeps the picked source and ticked axes across a list reload', async () => {
    const { admin, setRows } = await open();
    const select = q<HTMLSelectElement>('clone-settings-from')!;
    select.value = 'beta';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    const axis = q<HTMLInputElement>('clone-settings-axis-classes')!;
    axis.checked = false;
    axis.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();

    setRows([DEFAULT, ALPHA, BETA, testProject({ slug: 'gamma' })]);
    await admin.load();
    flushSync();

    expect(q<HTMLSelectElement>('clone-settings-from')!.value).toBe('beta');
    expect(q<HTMLInputElement>('clone-settings-axis-classes')!.checked).toBe(false);
  });

  it('does not reset the form during an in-flight submit', async () => {
    let release: (r: Response) => void = () => {};
    const { admin, setRows } = await open(
      undefined,
      () => new Promise<Response>((res) => (release = res)),
    );
    const select = q<HTMLSelectElement>('clone-settings-from')!;
    select.value = 'default';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    q('clone-settings-submit')!.closest('form')!.requestSubmit();
    await vi.waitFor(() =>
      expect(q('clone-settings-submit')!.textContent).toContain('Copying'),
    );

    setRows([DEFAULT, ALPHA, BETA, testProject({ slug: 'gamma' })]);
    await admin.load();
    flushSync();
    expect(q<HTMLSelectElement>('clone-settings-from')!.value).toBe('default');
    expect(q('clone-settings-submit')!.textContent).toContain('Copying');
    release(json({ project: ALPHA, warnings: [] }));
  });
});
