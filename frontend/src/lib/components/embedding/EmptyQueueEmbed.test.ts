import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { API_PREFIX } from '$lib/api';
import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
import { formatsFixture, reprocessFixture } from '$lib/test/fixtures/datasetImport';
import type { ReprocessRequest } from '$lib/types_import';
import EmptyQueueEmbed from './EmptyQueueEmbed.svelte';
import { emptyQueueEmbedRequest } from './emptyQueueEmbed';

const REQUEST: ReprocessRequest = {
  targets: { filter: { embedding_state: ['deferred', 'not_selected'] } },
  scopes: ['embed'],
  embed: { only_missing: true },
  dry_run: true,
};

const STATE = { has_unembedded_items: true, suggested_reprocess: REQUEST };
const REASON = 'no embedded items: embed them to rank this queue';

describe('emptyQueueEmbedRequest', () => {
  it('returns the served request only with the flag, the request and a vector reason', () => {
    expect(emptyQueueEmbedRequest(STATE, REASON)).toEqual(REQUEST);
    expect(
      emptyQueueEmbedRequest({ ...STATE, has_unembedded_items: false }, REASON),
    ).toBeNull();
    expect(
      emptyQueueEmbedRequest({ ...STATE, suggested_reprocess: null }, REASON),
    ).toBeNull();
    expect(emptyQueueEmbedRequest(STATE, 'no probe predictions: run a probe')).toBeNull();
    expect(emptyQueueEmbedRequest(null, REASON)).toBeNull();
    expect(emptyQueueEmbedRequest(STATE, null)).toBeNull();
  });
});

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;
let posts: unknown[];

beforeEach(() => {
  datasetsAvailability.reset();
  posts = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const u = String(url);
      const ok = (b: unknown) =>
        new Response(JSON.stringify(b), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        });
      if (u === `${API_PREFIX}/datasets/formats`) return ok(formatsFixture());
      posts.push(JSON.parse(String(init.body)));
      return ok(reprocessFixture());
    }),
  );
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
  datasetsAvailability.reset();
});

async function render(emptyState: typeof STATE | null, reasonText: string | null) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(EmptyQueueEmbed, { target, props: { emptyState, reasonText } });
  await datasetsAvailability.init();
  flushSync();
}

describe('EmptyQueueEmbed', () => {
  it('opens the Reprocess dialog with the served request, as served', async () => {
    await render(STATE, REASON);
    const open = target.querySelector<HTMLElement>('[data-testid="reprocess-open"]')!;
    expect(open.textContent?.trim()).toBe('Embed them');
    open.click();
    flushSync();
    [...document.querySelectorAll('button')]
      .find((b) => b.textContent?.trim() === 'Check what would run')!
      .click();
    await vi.waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0]).toEqual(REQUEST);
  });

  it('renders nothing without a served request for this reason', async () => {
    await render(STATE, 'no probe predictions: run a probe');
    expect(target.querySelector('[data-testid="empty-queue-embed"]')).toBeNull();
  });
});
