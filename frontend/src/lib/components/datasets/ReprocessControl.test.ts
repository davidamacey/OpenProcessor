/**
 * Reprocess (§7.12 item 6, W10.13), mounted: absent without the served
 * vocabulary; one crop applies from the dialog and hands back the served
 * post-write crop; several crops run a served dry run before the apply;
 * the served message and counts are what the operator reads.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { API_PREFIX } from '$lib/api';
import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
import { formatsFixture, reprocessFixture } from '$lib/test/fixtures/datasetImport';
import type { ReprocessTarget } from '$lib/datasets/reprocessController.svelte';
import ReprocessControl from './ReprocessControl.svelte';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;
let posts: Array<{ url: string; body: unknown }>;

function serve(routes: Record<string, (body: unknown) => Response>) {
  posts = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const u = String(url);
      if (u === `${API_PREFIX}/datasets/formats`) return routes.formats!(null);
      const body = init.body ? JSON.parse(String(init.body)) : undefined;
      posts.push({ url: u, body });
      for (const [k, fn] of Object.entries(routes)) if (u.endsWith(k)) return fn(body);
      return json({}, 404);
    }),
  );
}

async function render(t: ReprocessTarget, props: Record<string, unknown> = {}) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ReprocessControl, { target, props: { target: t, ...props } });
  await datasetsAvailability.init();
  flushSync();
}

function click(el: Element | null | undefined) {
  (el as HTMLElement).click();
  flushSync();
}

function buttonNamed(name: string): HTMLButtonElement | undefined {
  return [...document.querySelectorAll('button')].find(
    (b) => b.textContent?.trim() === name,
  );
}

function check(label: string) {
  const box = [...document.querySelectorAll('label')]
    .find((l) => l.textContent?.includes(label))
    ?.querySelector('input') as HTMLInputElement;
  box.checked = true;
  box.dispatchEvent(new Event('change', { bubbles: true }));
  flushSync();
}

const flush = () => new Promise((r) => setTimeout(r, 0));

beforeEach(() => datasetsAvailability.reset());
afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  vi.unstubAllGlobals();
  datasetsAvailability.reset();
});

describe('ReprocessControl', () => {
  it('is absent when the backend does not serve W10', async () => {
    serve({ formats: () => json({ detail: 'Not Found' }, 404) });
    await render({ kind: 'crop', cropId: 'c1' });
    expect(target.querySelector('[data-testid="reprocess-open"]')).toBeNull();
  });

  it('is absent when the served formats carry no reprocess vocabulary', async () => {
    serve({ formats: () => json(formatsFixture({ reprocess: null })) });
    await render({ kind: 'crop', cropId: 'c1' });
    expect(target.querySelector('[data-testid="reprocess-open"]')).toBeNull();
  });

  it('one crop: the served scopes and lock rule, then apply and adopt the served crop', async () => {
    serve({
      formats: () => json(formatsFixture()),
      '/crops/c1/reprocess': () =>
        json(
          reprocessFixture({
            dry_run: false,
            scopes: [{ scope: 'vlm', selected: 1, locked_skipped: 1, queued: 0 }],
            items: [{ crop_id: 'c1', class_name: 'widget' }],
            message: 'This item is locked; nothing was re-run.',
          }),
        ),
    });
    const onadopt = vi.fn();
    await render({ kind: 'crop', cropId: 'c1' }, { onadopt });
    click(target.querySelector('[data-testid="reprocess-open"]'));
    expect(
      document.querySelector('[data-testid="reprocess-lock-rule"]')?.textContent,
    ).toBe(formatsFixture().reprocess!.lock_rule);
    expect(buttonNamed('Reprocess')?.disabled).toBe(true);
    check('VLM class');
    click(buttonNamed('Reprocess'));
    await flush();
    flushSync();
    expect(posts).toEqual([
      {
        url: `${API_PREFIX}/crops/c1/reprocess`,
        body: { scopes: ['vlm'], dry_run: false },
      },
    ]);
    expect(document.querySelector('[data-testid="reprocess-message"]')?.textContent).toBe(
      'This item is locked; nothing was re-run.',
    );
    expect(onadopt).toHaveBeenCalledWith([expect.objectContaining({ id: 'c1' })]);
  });

  it('several crops: a served dry run first, then the apply', async () => {
    serve({
      formats: () => json(formatsFixture()),
      '/reprocess': (body) =>
        json(
          (body as { dry_run: boolean }).dry_run
            ? reprocessFixture()
            : reprocessFixture({
                dry_run: false,
                scopes: [{ scope: 'region', selected: 12, locked_skipped: 3, queued: 9 }],
                message: '9 items queued.',
              }),
        ),
    });
    const onapplied = vi.fn();
    await render({ kind: 'crops', cropIds: ['a', 'b'] }, { onapplied });
    click(target.querySelector('[data-testid="reprocess-open"]'));
    check('Regions');
    const mode = document.querySelector('select') as HTMLSelectElement;
    mode.value = 'reverify';
    mode.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(buttonNamed('Reprocess')).toBeUndefined();
    click(buttonNamed('Check what would run'));
    await flush();
    flushSync();
    const dry = document.querySelector('[data-testid="reprocess-dry-run"]')!;
    expect(dry.textContent).toContain('12');
    expect(dry.textContent).toContain('3');
    expect(dry.textContent).toContain(
      '12 items selected; 3 are locked and will be skipped.',
    );
    click(buttonNamed('Reprocess'));
    await flush();
    flushSync();
    expect(posts.map((p) => p.body)).toEqual([
      {
        targets: { crop_ids: ['a', 'b'] },
        scopes: ['region'],
        region_mode: 'reverify',
        dry_run: true,
      },
      {
        targets: { crop_ids: ['a', 'b'] },
        scopes: ['region'],
        region_mode: 'reverify',
        dry_run: false,
      },
    ]);
    expect(
      document.querySelector('[data-testid="reprocess-result"]')?.textContent,
    ).toContain('9 items queued.');
    expect(onapplied).toHaveBeenCalled();
  });

  it('a refusal shows the served message', async () => {
    serve({
      formats: () => json(formatsFixture()),
      '/reprocess': () =>
        json(
          {
            detail: {
              error: 'reprocess_busy',
              message: 'A reprocess job is already running.',
            },
          },
          409,
        ),
    });
    await render({ kind: 'crops', cropIds: ['a'] });
    click(target.querySelector('[data-testid="reprocess-open"]'));
    check('Embeddings');
    click(buttonNamed('Check what would run'));
    await flush();
    flushSync();
    expect(document.querySelector('[data-testid="reprocess-error"]')?.textContent).toBe(
      'A reprocess job is already running.',
    );
  });
});
