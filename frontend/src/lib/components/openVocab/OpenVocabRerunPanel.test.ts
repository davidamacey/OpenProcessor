/**
 * "Run on existing images": present only while a set is active and the
 * backend serves reprocess; the request is the all-images open_vocab one,
 * a served dry run comes first and the apply is a separate confirm.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { API_PREFIX } from '$lib/api';
import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
import { formatsFixture, reprocessFixture } from '$lib/test/fixtures/datasetImport';
import OpenVocabRerunPanel from './OpenVocabRerunPanel.svelte';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;
let posts: Array<{ url: string; body: unknown }>;

function serve(reprocess: (body: unknown) => Response, formatsStatus = 200) {
  posts = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const u = String(url);
      if (u === `${API_PREFIX}/datasets/formats`)
        return formatsStatus === 200 ? json(formatsFixture()) : json({}, formatsStatus);
      const body = init.body ? JSON.parse(String(init.body)) : undefined;
      posts.push({ url: u, body });
      return reprocess(body);
    }),
  );
}

async function render(active: boolean) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(OpenVocabRerunPanel, { target, props: { active } });
  await datasetsAvailability.init();
  flushSync();
}

const flush = () => new Promise((r) => setTimeout(r, 0));
const buttonNamed = (name: string) =>
  [...document.querySelectorAll('button')].find((b) => b.textContent?.trim() === name);

beforeEach(() => datasetsAvailability.reset());
afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  vi.unstubAllGlobals();
  datasetsAvailability.reset();
});

describe('OpenVocabRerunPanel', () => {
  it('is absent while no set is active', async () => {
    serve(() => json({}));
    await render(false);
    expect(target.querySelector('[data-testid="open-vocab-rerun"]')).toBeNull();
  });

  it('is absent when the backend does not serve reprocess', async () => {
    serve(() => json({}), 404);
    await render(true);
    expect(target.querySelector('[data-testid="open-vocab-rerun"]')).toBeNull();
  });

  it('dry-runs the all-images open_vocab request first, then applies it on confirm', async () => {
    serve((body) =>
      json(
        reprocessFixture({
          dry_run: (body as { dry_run: boolean }).dry_run,
          scopes: [
            {
              scope: 'open_vocab',
              selected: 12,
              queued: 0,
              detail: { estimated_calls: 24, segmenter_reachable: true },
            },
          ],
        }),
      ),
    );
    await render(true);
    (target.querySelector('[data-testid="reprocess-open"]') as HTMLElement).click();
    flushSync();
    buttonNamed('Check what would run')!.click();
    await flush();
    flushSync();
    expect(posts).toHaveLength(1);
    expect(posts[0]!.body).toEqual({
      targets: { filter: { all_images: true } },
      scopes: ['open_vocab'],
      dry_run: true,
    });
    buttonNamed('Reprocess')!.click();
    await flush();
    flushSync();
    expect(posts).toHaveLength(2);
    expect((posts[1]!.body as { dry_run: boolean }).dry_run).toBe(false);
  });
});
