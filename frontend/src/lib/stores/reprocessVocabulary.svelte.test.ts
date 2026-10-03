/**
 * reprocessVocabularyStore: the served `reprocess` block of
 * GET /config/vocabulary, read once per project. An id the vocabulary does
 * not list prints as served, never a guessed label.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { reprocessVocabularyFixture } from '$lib/test/fixtures/regionProfiles';
import { reprocessVocabularyStore as store } from './reprocessVocabulary.svelte';

const ok = (body: unknown) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });

beforeEach(() => store.resetForProjectChange());
afterEach(() => vi.unstubAllGlobals());

describe('reprocessVocabularyStore', () => {
  it('loads once and labels from the served entries', async () => {
    const f = vi.fn(async () => ok({ reprocess: reprocessVocabularyFixture() }));
    vi.stubGlobal('fetch', f);
    await Promise.all([store.init(), store.init()]);
    await store.init();
    expect(f).toHaveBeenCalledTimes(1);
    expect(store.label('scopes', 'embed')).toBe('Compute vectors');
    expect(store.description('scopes', 'embed')).toBe(
      'Compute vectors: served description',
    );
    expect(store.lockText()).toContain('Human label');
    expect(store.lockText()).toContain('Test holdout: served description');
  });

  it('prints an unlisted id as served, not humanized', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ok({ reprocess: reprocessVocabularyFixture() })),
    );
    await store.init();
    expect(store.label('scopes', 'brand_new_scope')).toBe('brand_new_scope');
  });

  it('when the read fails, ids print as served and lock text is bare', async () => {
    expect(store.label('scopes', 'detect')).toBe('detect');
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('{}', { status: 500 })),
    );
    await store.init();
    expect(store.label('scopes', 'detect')).toBe('detect');
    expect(store.lockText()).toBe('Locked');
  });

  it('forgets everything on a project change', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ok({ reprocess: reprocessVocabularyFixture() })),
    );
    await store.init();
    store.resetForProjectChange();
    expect(store.label('scopes', 'embed')).toBe('embed');
  });
});
