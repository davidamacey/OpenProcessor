import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { API_PREFIX } from '$lib/api';
import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
import { formatsFixture, reprocessFixture } from '$lib/test/fixtures/datasetImport';
import VectorRefreshNotice from './VectorRefreshNotice.svelte';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;
const posts: Array<{ url: string; body: unknown }> = [];

beforeEach(() => {
  posts.length = 0;
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const u = String(url);
      if (u === `${API_PREFIX}/datasets/formats`) return json(formatsFixture());
      posts.push({ url: u, body: init.body ? JSON.parse(String(init.body)) : undefined });
      if (u.endsWith('/reprocess')) return json(reprocessFixture());
      return json({}, 404);
    }),
  );
  target = document.createElement('div');
  document.body.appendChild(target);
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target.remove();
  document.querySelectorAll('[role="dialog"]').forEach((d) => d.remove());
  vi.unstubAllGlobals();
});

async function render(refresh: { embedded: number; pending: number } | null) {
  instance = mount(VectorRefreshNotice, { target, props: { cropId: 'c1', refresh } });
  await datasetsAvailability.init();
  flushSync();
}

describe('VectorRefreshNotice', () => {
  it('renders nothing for a null or zero-pending refresh', async () => {
    await render(null);
    expect(target.querySelector('[data-testid="vector-refresh-notice"]')).toBeNull();
    unmount(instance!);
    await render({ embedded: 3, pending: 0 });
    expect(target.querySelector('[data-testid="vector-refresh-notice"]')).toBeNull();
  });

  it('shows the served pending count and an Embed now button', async () => {
    await render({ embedded: 1, pending: 2 });
    const notice = target.querySelector('[data-testid="vector-refresh-notice"]')!;
    expect(notice.textContent?.replace(/\s+/g, ' ')).toContain(
      '2 boxes have no vector yet',
    );
    expect(notice.textContent).toContain('Embed now');
  });

  it('Embed now opens a Reprocess dialog whose dry run asks for the embed scope, missing vectors only', async () => {
    await render({ embedded: 0, pending: 1 });
    expect(target.textContent?.replace(/\s+/g, ' ')).toContain('1 box has no vector yet');
    (target.querySelector('[data-testid="reprocess-open"]') as HTMLButtonElement).click();
    flushSync();
    const dialog = document.querySelector('[role="dialog"]')!;
    expect(dialog).not.toBeNull();
    const check = [...dialog.querySelectorAll('button')].find((b) =>
      b.textContent?.includes('Check what would run'),
    )!;
    check.click();
    await new Promise((r) => setTimeout(r, 0));
    expect(posts.find((p) => p.url.endsWith('/reprocess'))?.body).toEqual({
      targets: { crop_ids: ['c1'] },
      scopes: ['embed'],
      embed: { only_missing: true },
      dry_run: true,
    });
  });
});
