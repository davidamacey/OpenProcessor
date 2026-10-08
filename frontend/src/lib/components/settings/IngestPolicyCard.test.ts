import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import IngestPolicyCard from './IngestPolicyCard.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  vi.unstubAllGlobals();
});

function render(make: () => Response) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => make()),
  );
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(IngestPolicyCard, { target });
  flushSync();
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });

describe('IngestPolicyCard', () => {
  it('links to the editor and shows the served embedding mode', async () => {
    render(() => json({ embedding: { mode: 'lazy' }, revision: 1 }));
    await vi.waitFor(() => {
      flushSync();
      expect(
        target.querySelector('[data-testid="ingest-policy-card-mode"]')?.textContent,
      ).toContain('Embedding: lazy');
    });
    expect(target.querySelector('a[href$="/settings/ingest-policy"]')).not.toBeNull();
  });

  it('shows the served error when the read fails, link kept', async () => {
    render(() => json({ detail: 'store down' }, 500));
    await vi.waitFor(
      () => {
        flushSync();
        expect(
          target.querySelector('[data-testid="ingest-policy-card-error"]')?.textContent,
        ).toContain('store down');
      },
      { timeout: 4000 },
    );
    expect(target.querySelector('a[href$="/settings/ingest-policy"]')).not.toBeNull();
  });
});
