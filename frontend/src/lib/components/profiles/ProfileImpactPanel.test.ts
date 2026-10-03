/**
 * The activation impact, mounted: the served counts and the by-profile
 * table; "Re-run" only when the server suggests one AND serves W10's
 * Reprocess; the served request goes out as served, dry run first, and
 * the apply only after a confirm; the served counts are what
 * the operator reads.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { API_PREFIX } from '$lib/api';
import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
import { formatsFixture, reprocessFixture } from '$lib/test/fixtures/datasetImport';
import {
  impactFixture,
  reprocessVocabularyFixture,
} from '$lib/test/fixtures/regionProfiles';
import type { ActivationImpact } from '$lib/types_profiles';
import { reprocessVocabularyStore } from '$lib/stores/reprocessVocabulary.svelte';
import ProfileImpactPanel from './ProfileImpactPanel.svelte';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;
let posts: Array<{ url: string; body: Record<string, unknown> }>;

function serve(formats: () => Response) {
  posts = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const u = String(url);
      if (u === `${API_PREFIX}/datasets/formats`) return formats();
      if (u === `${API_PREFIX}/config/vocabulary`)
        return json({ reprocess: reprocessVocabularyFixture() });
      const body = init.body ? JSON.parse(String(init.body)) : undefined;
      posts.push({ url: u, body });
      if (u === `${API_PREFIX}/reprocess`) {
        return json(
          body.dry_run
            ? reprocessFixture()
            : reprocessFixture({
                dry_run: false,
                scopes: [{ scope: 'region', selected: 12, locked_skipped: 3, queued: 9 }],
              }),
        );
      }
      return json({}, 404);
    }),
  );
}

async function render(impact: ActivationImpact, probe = true) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ProfileImpactPanel, { target, props: { impact } });
  if (probe) await datasetsAvailability.init();
  if (probe) await reprocessVocabularyStore.init();
  flushSync();
}

const q = (id: string) => document.querySelector<HTMLElement>(`[data-testid="${id}"]`);
beforeEach(() => {
  datasetsAvailability.reset();
  reprocessVocabularyStore.resetForProjectChange();
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  vi.unstubAllGlobals();
  datasetsAvailability.reset();
  document.querySelectorAll('[role="dialog"]').forEach((d) => d.remove());
});

describe('ProfileImpactPanel', () => {
  it('shows the served counts and which profile produced what', async () => {
    serve(() => json(formatsFixture()));
    await render(impactFixture());
    expect(q('impact-items_total')?.textContent?.trim()).toBe((1840).toLocaleString());
    expect(q('impact-unseeded_items')?.textContent?.trim()).toBe('850');
    expect(q('impact-by-profile')?.textContent).toContain('env_tags');
    expect(q('impact-by-profile')?.textContent).toContain('widget_tag r3');
  });

  it('no Re-run without a served suggestion (and no W10 probe)', async () => {
    serve(() => json(formatsFixture()));
    await render(impactFixture({ suggested_reprocess: null }), false);
    expect(q('rerun-open')).toBeNull();
    expect(vi.mocked(fetch)).not.toHaveBeenCalled();
  });

  it('no Re-run without a served suggestion even when Reprocess is served', async () => {
    serve(() => json(formatsFixture()));
    await render(impactFixture({ suggested_reprocess: null }));
    expect(q('profile-impact')).not.toBeNull();
    expect(q('rerun-open')).toBeNull();
  });

  it('no Re-run when the backend does not serve Reprocess', async () => {
    serve(() => json({ detail: 'Not Found' }, 404));
    await render(impactFixture());
    expect(q('rerun-open')).toBeNull();
  });

  it('Re-run: the served request as served, dry run first, apply only after the confirm', async () => {
    serve(() => json(formatsFixture()));
    const impact = impactFixture();
    await render(impact);
    q('rerun-open')!.click();
    await vi.waitFor(() => expect(q('rerun-dry-run')).not.toBeNull());
    expect(posts).toEqual([
      {
        url: `${API_PREFIX}/reprocess`,
        body: { ...impact.suggested_reprocess, dry_run: true },
      },
    ]);
    expect(q('rerun-dry-run')?.textContent).toContain('Region stage');
    expect(q('rerun-dry-run')?.textContent).toContain('12');
    q('rerun-apply')!.click();
    flushSync();
    expect(posts).toHaveLength(1);
    const dialog = document.querySelector('[role="dialog"]')!;
    expect(dialog.getAttribute('aria-label')).toBe('Re-run items');
    [...dialog.querySelectorAll('button')]
      .find((b) => b.textContent?.trim() === 'Re-run')!
      .click();
    await vi.waitFor(() => expect(q('rerun-result')).not.toBeNull());
    expect(posts[1]).toEqual({
      url: `${API_PREFIX}/reprocess`,
      body: { ...impact.suggested_reprocess, dry_run: false },
    });
    expect(q('rerun-result')?.textContent).toContain('Region');
    expect(
      [...q('rerun-result')!.querySelectorAll('tbody tr td')].map((c) =>
        c.textContent?.trim(),
      ),
    ).toEqual(['Region stage', '12', '3', '9', '—', '—']);
    expect(document.querySelector('[role="dialog"]')).toBeNull();
  });
});
