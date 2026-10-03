/**
 * The region-stage panel, mounted: the served state and counts, a confirm
 * before any write, the served state adopted after it, the served refusal
 * text, and the re-run request being exactly the served one.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { API_PREFIX } from '$lib/api';
import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
import { formatsFixture, reprocessFixture } from '$lib/test/fixtures/datasetImport';
import type { RegionStageState } from '$lib/types_openVocab';
import RegionStagePanel from './RegionStagePanel.svelte';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

const RERUN = {
  targets: { filter: { region_gate_skipped: true } },
  scopes: ['region'],
  dry_run: true,
};

const stage = (over: Partial<RegionStageState> = {}): RegionStageState =>
  ({
    project: 'alpha',
    paused: false,
    paused_since: null,
    pipeline_paused: false,
    counts: { pending_detection: 4, pending_verification: 1, gate_skipped: 7 },
    rerun_skipped: RERUN,
    ...over,
  }) as RegionStageState;

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;
let calls: Array<{ method: string; url: string; body: unknown }>;

function serve(routes: Record<string, () => Response>) {
  calls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const u = String(url);
      const method = init.method ?? 'GET';
      if (u === `${API_PREFIX}/datasets/formats`) return json(formatsFixture());
      calls.push({
        method,
        url: u,
        body: init.body ? JSON.parse(String(init.body)) : undefined,
      });
      for (const [k, fn] of Object.entries(routes)) {
        const [m, path] = k.split(' ');
        if (m === method && u.endsWith(path!)) return fn();
      }
      return json({}, 404);
    }),
  );
}

async function render() {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(RegionStagePanel, { target });
  await datasetsAvailability.init();
  await flush();
  flushSync();
}

const flush = () => new Promise((r) => setTimeout(r, 0));
const q = (id: string) =>
  target.querySelector<HTMLElement>(`[data-testid="${id}"]`) as HTMLElement;
const buttonNamed = (name: string) =>
  [...document.querySelectorAll<HTMLElement>('[role="dialog"] button')].find(
    (b) => b.textContent?.trim() === name,
  );

beforeEach(() => datasetsAvailability.reset());
afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  vi.unstubAllGlobals();
  datasetsAvailability.reset();
});

describe('RegionStagePanel', () => {
  it('shows the served state and counts', async () => {
    serve({ 'GET /region_stage': () => json(stage()) });
    await render();
    expect(q('region-stage-state').textContent).toBe('running');
    expect(q('region-stage-pending-detection').textContent?.trim()).toBe('4');
    expect(q('region-stage-pending-verification').textContent?.trim()).toBe('1');
    expect(q('region-stage-gate-skipped').textContent?.trim()).toBe('7');
    expect(q('region-stage-pipeline-paused')).toBeNull();
  });

  it('names the paused state, since when, and a paused pipeline', async () => {
    serve({
      'GET /region_stage': () =>
        json(
          stage({
            paused: true,
            paused_since: '2026-10-03T08:00:00Z',
            pipeline_paused: true,
          }),
        ),
    });
    await render();
    expect(q('region-stage-state').textContent).toBe('paused');
    expect(q('region-stage-since')).not.toBeNull();
    expect(q('region-stage-pipeline-paused')).not.toBeNull();
    expect(q('region-stage-toggle').textContent?.trim()).toBe('Resume');
  });

  it('writes nothing until Pause is confirmed, then shows the served state', async () => {
    serve({
      'GET /region_stage': () => json(stage()),
      'POST /region_stage/pause': () => json(stage({ paused: true })),
    });
    await render();
    q('region-stage-toggle').click();
    flushSync();
    expect(calls.some((c) => c.method === 'POST')).toBe(false);
    expect(document.body.textContent).toContain('Queued items stay pending');
    buttonNamed('Pause')!.click();
    await flush();
    flushSync();
    expect(calls.filter((c) => c.method === 'POST')).toHaveLength(1);
    expect(q('region-stage-state').textContent).toBe('paused');
  });

  it('cancelling the confirm sends nothing', async () => {
    serve({ 'GET /region_stage': () => json(stage()) });
    await render();
    q('region-stage-toggle').click();
    flushSync();
    buttonNamed('Cancel')!.click();
    flushSync();
    expect(calls.some((c) => c.method === 'POST')).toBe(false);
  });

  it('shows a refusal as served', async () => {
    serve({
      'GET /region_stage': () => json(stage()),
      'POST /region_stage/pause': () => json({ detail: 'Stage is locked.' }, 409),
    });
    await render();
    q('region-stage-toggle').click();
    flushSync();
    buttonNamed('Pause')!.click();
    await flush();
    flushSync();
    expect(
      document.querySelector('[data-testid="region-stage-action-error"]')!.textContent,
    ).toBe('Stage is locked.');
  });

  it('re-runs gate-skipped with the served request, as served', async () => {
    serve({
      'GET /region_stage': () => json(stage()),
      'POST /reprocess': () =>
        json(
          reprocessFixture({ dry_run: true, scopes: [{ scope: 'region', selected: 7 }] }),
        ),
    });
    await render();
    expect((q('reprocess-open') as HTMLElement).textContent?.trim()).toBe(
      'Re-run gate-skipped (7)…',
    );
    q('reprocess-open').click();
    flushSync();
    buttonNamed('Check what would run')!.click();
    await flush();
    flushSync();
    const post = calls.find((c) => c.method === 'POST')!;
    expect(post.body).toEqual(RERUN);
  });

  it('offers no re-run when nothing was gate-skipped', async () => {
    serve({
      'GET /region_stage': () =>
        json(
          stage({
            counts: { pending_detection: 0, pending_verification: 0, gate_skipped: 0 },
          }),
        ),
    });
    await render();
    expect(q('reprocess-open')).toBeNull();
  });

  it('shows a failed read as an error', async () => {
    serve({ 'GET /region_stage': () => json({ detail: 'boom' }, 400) });
    await render();
    expect(q('region-stage-error').textContent).toContain('boom');
  });
});
