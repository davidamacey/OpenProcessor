import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { API_PREFIX } from '$lib/api';
import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
import { formatsFixture } from '$lib/test/fixtures/datasetImport';
import type { ReprocessRequest } from '$lib/types_import';
import UnembeddedBanner from './UnembeddedBanner.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

beforeEach(() => {
  datasetsAvailability.reset();
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) =>
      String(url) === `${API_PREFIX}/datasets/formats`
        ? new Response(JSON.stringify(formatsFixture()), {
            status: 200,
            headers: { 'content-type': 'application/json' },
          })
        : new Response('{}', { status: 404 }),
    ),
  );
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  vi.unstubAllGlobals();
  datasetsAvailability.reset();
});

const REQUEST: ReprocessRequest = {
  targets: { filter: { embedding_state: ['deferred'] } },
  scopes: ['embed'],
  dry_run: true,
};

async function render(props: {
  count: number | null | undefined;
  context: 'ordered' | 'search';
  suggestedReprocess?: ReprocessRequest | null;
}) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(UnembeddedBanner, { target, props });
  await datasetsAvailability.init();
  flushSync();
}

const banner = () => target.querySelector('[data-testid="unembedded-banner"]');

describe('UnembeddedBanner', () => {
  it('is absent for 0, null and undefined', async () => {
    for (const count of [0, null, undefined]) {
      await render({ count, context: 'ordered' });
      expect(banner()).toBeNull();
      unmount(instance!);
      target.remove();
    }
    instance = undefined;
  });

  it('says the items are not ranked in an ordered view', async () => {
    await render({ count: 1234, context: 'ordered' });
    expect(banner()!.textContent).toContain(
      '1,234 items in scope have no vector and are not ranked',
    );
  });

  it('says the items cannot match a text search', async () => {
    await render({ count: 7, context: 'search' });
    expect(banner()!.textContent).toContain(
      '7 items in scope have no vector and cannot match a text search',
    );
  });

  it('offers "Embed them" only with a served request, passed as served', async () => {
    await render({ count: 5, context: 'ordered' });
    expect(target.querySelector('[data-testid="reprocess-open"]')).toBeNull();
    unmount(instance!);
    target.remove();
    await render({ count: 5, context: 'ordered', suggestedReprocess: REQUEST });
    expect(
      target.querySelector('[data-testid="reprocess-open"]')!.textContent?.trim(),
    ).toBe('Embed them');
  });
});
