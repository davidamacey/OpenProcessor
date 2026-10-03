/** The `/settings` card is absent (not disabled) unless the sets are served. */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { listFixture } from '$lib/openVocab/fixtures';
import { openVocabAvailability } from '$lib/openVocab/openVocabAvailability.svelte';
import OpenVocabCard from './OpenVocabCard.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

async function render(status: number) {
  vi.stubGlobal(
    'fetch',
    vi.fn(
      async () =>
        new Response(JSON.stringify(status === 200 ? listFixture() : { detail: 'x' }), {
          status,
          headers: { 'content-type': 'application/json' },
        }),
    ),
  );
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(OpenVocabCard, { target });
  await openVocabAvailability.init();
  flushSync();
}

beforeEach(() => openVocabAvailability.reset());
afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  vi.unstubAllGlobals();
  openVocabAvailability.reset();
});

describe('OpenVocabCard', () => {
  it('links to the editor when served', async () => {
    await render(200);
    const a = target.querySelector('a')!;
    expect(a.getAttribute('href')).toContain('/settings/open-vocab');
  });

  it('renders nothing when the backend does not serve the routes', async () => {
    await render(404);
    expect(target.querySelector('[data-testid="open-vocab-card"]')).toBeNull();
  });
});
