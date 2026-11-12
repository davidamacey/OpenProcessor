/**
 * The Settings layout: a sidebar of the sections the backend serves (absent, not disabled, for a
 * feature it does not), a search over the static index that `/` focuses, and the scope badge.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';
import { mountUi, snip } from '$lib/test/mountUi';

const state = vi.hoisted(() => ({ page: { url: new URL('http://x/p/demo/settings') } }));
vi.mock('$app/state', () => state);
vi.mock('$lib/projectPaths', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/projectPaths')>('$lib/projectPaths');
  return {
    ...actual,
    projectHref: (path: string, slug = 'demo') => `/p/${slug}${path}`,
  };
});

const { default: Layout } = await import('./+layout.svelte');
const { packsAvailability } = await import('$lib/packs/packsAvailability.svelte');
const { profilesAvailability } =
  await import('$lib/profiles/profilesAvailability.svelte');
const { vlmAvailability } = await import('$lib/vlm/vlmAvailability.svelte');
const { openVocabAvailability } =
  await import('$lib/openVocab/openVocabAvailability.svelte');
const { keymapAvailability } = await import('$stores/keymap.svelte');

const stores = [
  packsAvailability,
  profilesAvailability,
  vlmAvailability,
  openVocabAvailability,
] as const;

function serve(served: boolean): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () =>
      served
        ? new Response(JSON.stringify({ items: [], packs: [], profiles: [], sets: [] }), {
            status: 200,
            headers: { 'content-type': 'application/json' },
          })
        : new Response(JSON.stringify({ detail: 'Not Found' }), {
            status: 404,
            headers: { 'content-type': 'application/json' },
          }),
    ),
  );
}

const names = (el: HTMLElement) =>
  [...el.querySelectorAll('[data-testid="sidebar-nav"] a')].map((a) =>
    a.textContent?.trim(),
  );

beforeEach(() => {
  state.page.url = new URL('http://x/p/demo/settings');
});
afterEach(() => {
  vi.unstubAllGlobals();
  for (const s of stores) s.reset();
});

async function settle(): Promise<void> {
  await new Promise((r) => setTimeout(r, 30));
  flushSync();
}

describe('Settings layout', () => {
  it('without the optional features the sidebar has only the always-served sections', async () => {
    serve(false);
    const el = mountUi(Layout, { children: snip('<p>body</p>') });
    await settle();
    expect(names(el)).toEqual(
      expect.arrayContaining([
        'General',
        'Ingest policy',
        'Embedding spaces',
        'Cluster profiles',
        'Curation scores',
        'Serving profiles',
        'Keyboard shortcuts',
      ]),
    );
    for (const absent of [
      'Prompt packs',
      'Region profiles',
      'Models and VLM',
      'Open vocabulary',
    ])
      expect(names(el)).not.toContain(absent);
  });

  it('a served feature adds its section', async () => {
    serve(true);
    const el = mountUi(Layout, { children: snip('<p>body</p>') });
    await settle();
    expect(names(el)).toEqual(
      expect.arrayContaining(['Prompt packs', 'Region profiles', 'Models and VLM']),
    );
  });

  it('marks the current section and shows its scope', async () => {
    serve(false);
    state.page.url = new URL('http://x/p/demo/settings/presets/serving-profiles');
    const el = mountUi(Layout, { children: snip('<p>body</p>') });
    await settle();
    expect(el.querySelector('a[aria-current="page"]')?.textContent).toContain(
      'Serving profiles',
    );
    expect(el.querySelector('[data-testid="settings-content"]')!.textContent).toContain(
      'Deployment',
    );
  });

  it('the search filters the sections and "/" focuses it', async () => {
    serve(false);
    const el = mountUi(Layout, { children: snip('<p>body</p>') });
    await settle();
    window.dispatchEvent(new KeyboardEvent('keydown', { key: '/', cancelable: true }));
    const input = el.querySelector<HTMLInputElement>('[data-testid="settings-search"]')!;
    expect(document.activeElement).toBe(input);
    input.value = 'cluster';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(names(el)).toEqual(['Cluster profiles']);
    input.value = 'zzzz';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(el.querySelector('[data-testid="settings-search-empty"]')).not.toBeNull();
  });

  it('"/" typed in a field does not steal focus', async () => {
    serve(false);
    const el = mountUi(Layout, {
      children: snip('<input data-testid="other" />'),
    });
    await settle();
    const other = el.querySelector<HTMLInputElement>('[data-testid="other"]')!;
    other.focus();
    other.dispatchEvent(new KeyboardEvent('keydown', { key: '/', bubbles: true }));
    expect(document.activeElement).toBe(other);
  });
});

void keymapAvailability;
