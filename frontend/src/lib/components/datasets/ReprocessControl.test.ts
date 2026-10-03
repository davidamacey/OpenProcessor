/**
 * Reprocess (§7.12 item 6, W10.13), mounted against the contract shapes:
 * present whenever W10 is served (the backend serves no `reprocess`
 * vocabulary block, so nothing is gated on one); the scope and region-mode
 * ids are the contract's enums; one crop or one image applies from the
 * dialog and hands back the served post-write items; several crops run a
 * served dry run before the apply; the served counts are what the operator
 * reads.
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

  it('is present once W10 is served, with no reprocess block in the formats', async () => {
    const formats = formatsFixture();
    expect(formats).not.toHaveProperty('reprocess');
    serve({ formats: () => json(formats) });
    await render({ kind: 'crop', cropId: 'c1' });
    expect(target.querySelector('[data-testid="reprocess-open"]')).not.toBeNull();
  });

  it('offers exactly the contract scopes and region modes, and no lock-rule copy', async () => {
    serve({ formats: () => json(formatsFixture()) });
    await render({ kind: 'crop', cropId: 'c1' });
    click(target.querySelector('[data-testid="reprocess-open"]'));
    const labels = [...document.querySelectorAll('fieldset label')].map((l) =>
      l.textContent?.trim(),
    );
    expect(labels).toEqual(['Detect', 'Open vocab', 'Region', 'VLM', 'Embed']);
    expect(document.querySelector('[data-testid="reprocess-lock-rule"]')).toBeNull();
    expect(document.querySelector('select')).toBeNull();
    check('Region');
    const modes = [...document.querySelectorAll('select option')].map(
      (o) => (o as HTMLOptionElement).value,
    );
    expect(modes).toEqual(['', 'redetect', 'reverify']);
  });

  it('one crop: apply with dry_run false and adopt the served crop', async () => {
    serve({
      formats: () => json(formatsFixture()),
      '/crops/c1/reprocess': () =>
        json(
          reprocessFixture({
            dry_run: false,
            scopes: [{ scope: 'vlm', selected: 1, locked_skipped: 1, queued: 0 }],
            items: [{ crop_id: 'c1', class_name: 'widget' }],
          }),
        ),
    });
    const onadopt = vi.fn();
    await render({ kind: 'crop', cropId: 'c1' }, { onadopt });
    click(target.querySelector('[data-testid="reprocess-open"]'));
    expect(buttonNamed('Reprocess')?.disabled).toBe(true);
    check('VLM');
    click(buttonNamed('Reprocess'));
    await flush();
    flushSync();
    expect(posts).toEqual([
      {
        url: `${API_PREFIX}/crops/c1/reprocess`,
        body: { scopes: ['vlm'], dry_run: false },
      },
    ]);
    const counts = document.querySelector('[data-testid="reprocess-result"]')!;
    expect(counts.textContent).toContain('Locked, skipped');
    expect(onadopt).toHaveBeenCalledWith([expect.objectContaining({ id: 'c1' })]);
  });

  it('one image: posts to the image route and adopts every served item', async () => {
    serve({
      formats: () => json(formatsFixture()),
      '/images/img_1/reprocess': () =>
        json(
          reprocessFixture({
            dry_run: false,
            items: [{ crop_id: 'c1' }, { crop_id: 'c2' }],
          }),
        ),
    });
    const onadopt = vi.fn();
    await render({ kind: 'image', imageId: 'img_1' }, { onadopt });
    click(target.querySelector('[data-testid="reprocess-open"]'));
    expect(document.querySelector('h3')?.textContent?.trim()).toBe('Reprocess image');
    check('Detect');
    click(buttonNamed('Reprocess'));
    await flush();
    flushSync();
    expect(posts).toEqual([
      {
        url: `${API_PREFIX}/images/img_1/reprocess`,
        body: { scopes: ['detect'], dry_run: false },
      },
    ]);
    expect(onadopt).toHaveBeenCalledWith([
      expect.objectContaining({ id: 'c1' }),
      expect.objectContaining({ id: 'c2' }),
    ]);
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
              }),
        ),
    });
    const onapplied = vi.fn();
    await render({ kind: 'crops', cropIds: ['a', 'b'] }, { onapplied });
    click(target.querySelector('[data-testid="reprocess-open"]'));
    check('Region');
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
    const cells = [
      ...document.querySelectorAll('[data-testid="reprocess-result"] tbody tr td'),
    ].map((c) => c.textContent?.trim());
    // scope, selected, locked, queued, failed, not found: omitted counts read "—".
    expect(cells).toEqual(['Region', '12', '3', '9', '—', '—']);
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
    check('Embed');
    click(buttonNamed('Check what would run'));
    await flush();
    flushSync();
    expect(document.querySelector('[data-testid="reprocess-error"]')?.textContent).toBe(
      'A reprocess job is already running.',
    );
  });

  it('shows a served job with its progress', async () => {
    serve({
      formats: () => json(formatsFixture()),
      '/reprocess': (body) =>
        json(
          (body as { dry_run: boolean }).dry_run
            ? reprocessFixture()
            : reprocessFixture({
                dry_run: false,
                job: {
                  job_id: 'rj1',
                  status: 'running',
                  images_total: 10,
                  images_done: 4,
                  images_failed: 1,
                  poll_after_s: 600,
                },
              }),
        ),
    });
    await render({ kind: 'crops', cropIds: ['a'] });
    click(target.querySelector('[data-testid="reprocess-open"]'));
    check('Detect');
    click(buttonNamed('Check what would run'));
    await flush();
    flushSync();
    click(buttonNamed('Reprocess'));
    await flush();
    flushSync();
    const job = document.querySelector('[data-testid="reprocess-job"]')!;
    expect(job.textContent).toContain('rj1');
    expect(job.textContent).toContain('4 / 10');
    expect(job.textContent).toContain('1 failed');
    click(buttonNamed('Close'));
  });

  it('batch: embed options appear with the embed scope, and are sent only once touched', async () => {
    serve({
      formats: () => json(formatsFixture()),
      '/reprocess': () => json(reprocessFixture()),
    });
    await render({ kind: 'crops', cropIds: ['a', 'b'] });
    click(target.querySelector('[data-testid="reprocess-open"]'));
    expect(document.querySelector('[data-testid="reprocess-embed-options"]')).toBeNull();
    check('Embed');
    expect(
      document.querySelector('[data-testid="reprocess-embed-options"]'),
    ).not.toBeNull();
    click(buttonNamed('Check what would run'));
    await flush();
    expect(posts.at(-1)!.body).not.toHaveProperty('embed');

    check('Only items without a vector');
    check('Frame');
    click(buttonNamed('Check what would run'));
    await flush();
    expect(posts.at(-1)!.body).toMatchObject({
      embed: { only_missing: true, parts: ['frame'] },
    });
  });

  it('one crop: the embed scope offers no embed options (the body has none)', async () => {
    serve({ formats: () => json(formatsFixture()) });
    await render({ kind: 'crop', cropId: 'c1' });
    click(target.querySelector('[data-testid="reprocess-open"]'));
    check('Embed');
    expect(document.querySelector('[data-testid="reprocess-embed-options"]')).toBeNull();
  });

  it('shows the served per-scope detail, booleans as yes/no', async () => {
    serve({
      formats: () => json(formatsFixture()),
      '/reprocess': () =>
        json(
          reprocessFixture({
            scopes: [
              {
                scope: 'embed',
                selected: 5,
                detail: {
                  to_embed: 4,
                  estimated_vector_kb: 12.5,
                  segmenter_reachable: false,
                },
              },
            ],
          }),
        ),
    });
    await render({ kind: 'crops', cropIds: ['a'] });
    click(target.querySelector('[data-testid="reprocess-open"]'));
    check('Embed');
    click(buttonNamed('Check what would run'));
    await flush();
    flushSync();
    const detail = document.querySelector('[data-testid="reprocess-detail"]')!;
    expect(detail.textContent).toContain('To embed');
    expect(detail.textContent).toContain('4');
    expect(detail.textContent).toContain('Estimated vector op');
    expect(detail.textContent).toContain('12.5');
    expect(detail.textContent).toMatch(/Segmenter reachable\s*no/);
  });
});
