/** `/settings` (General): a failed read is an ErrorState with the request id and a Retry; a
 *  backend that offers no settable default shows an empty state, never an empty card. */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import SettingsPage from './+page.svelte';
import { curationSettingsStore } from '$stores/curationSettings.svelte';
import { strategiesStore } from '$stores/strategies.svelte';

const ok = (body: unknown) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;
let settingsCalls: number;
let settingsFails: boolean;

beforeEach(() => {
  settingsCalls = 0;
  curationSettingsStore.reset();
  strategiesStore.reset();
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      const u = String(url);
      if (u.endsWith('/methods')) return ok({ strategies: [], flags: {} });
      if (u.endsWith('/settings')) {
        settingsCalls += 1;
        return settingsFails && settingsCalls === 1
          ? new Response(JSON.stringify({ detail: 'Not Found' }), {
              status: 404,
              headers: { 'content-type': 'application/json', 'x-request-id': 'req-3' },
            })
          : ok({ defaults: {}, revision: 1, updated_at: null, updated_by: null });
      }
      return new Response(JSON.stringify({ detail: 'Not Found' }), { status: 404 });
    }),
  );
});
afterEach(() => {
  if (instance) unmount(instance);
  target?.remove();
  vi.unstubAllGlobals();
});

async function render(): Promise<void> {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(SettingsPage, { target });
  await vi.waitFor(() => {
    flushSync();
    expect(
      target.querySelector('[data-testid="error-state"],[data-testid="empty-state"]'),
    ).not.toBeNull();
  });
}

describe('/settings General states', () => {
  it('a failed read is an ErrorState with the request id, and Retry reads again', async () => {
    settingsFails = true;
    await render();
    const err = target.querySelector('[data-testid="settings-load-error"]')!;
    expect(err.querySelector('[data-testid="error-state-request-id"]')?.textContent).toBe(
      'req-3',
    );
    [...err.querySelectorAll('button')]
      .find((b) => b.textContent?.includes('Retry'))!
      .click();
    await vi.waitFor(() => {
      flushSync();
      expect(target.querySelector('[data-testid="settings-load-error"]')).toBeNull();
    });
  });

  it('no settable default is one empty state, not an empty "Active defaults" card', async () => {
    settingsFails = false;
    await render();
    expect(target.querySelector('[data-testid="settings-no-defaults"]')).not.toBeNull();
    expect(target.textContent).not.toContain('Active defaults');
  });
});
