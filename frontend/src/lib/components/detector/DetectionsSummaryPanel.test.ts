import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { API_PREFIX } from '$lib/api';
import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
import { DetectionsSummaryState } from '$lib/detector/detectionsSummaryController.svelte';
import { formatsFixture, reprocessFixture } from '$lib/test/fixtures/datasetImport';
import type { DetectionsSummary } from '$lib/types_detector';
import DetectionsSummaryPanel from './DetectionsSummaryPanel.svelte';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

const REQUEST = {
  targets: { filter: { embedding_state: ['not_selected' as const, 'failed' as const] } },
  scopes: ['embed' as const],
  dry_run: true,
};

const SUMMARY: DetectionsSummary = {
  total: 120,
  embedding: {
    embedded: 100,
    not_embedded: 20,
    by_state: { embedded: 100, not_selected: 15, failed: 5 },
  },
  by_label: [
    {
      name: 'widget',
      count: 80,
      embedding: { embedded: 70, not_embedded: 10, by_state: {} },
    },
  ],
  labels_truncated: true,
  suggested_reprocess: REQUEST,
};

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;
let posts: { url: string; body: unknown }[];

function serve(summary: DetectionsSummary | null) {
  posts = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const u = String(url);
      if (u === `${API_PREFIX}/datasets/formats`) return json(formatsFixture());
      if (u.endsWith('/detections/summary'))
        return summary ? json(summary) : json({ detail: 'index down' }, 503);
      posts.push({ url: u, body: init.body ? JSON.parse(String(init.body)) : undefined });
      if (u.endsWith('/reprocess')) return json(reprocessFixture());
      return json({}, 404);
    }),
  );
}

async function render() {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(DetectionsSummaryPanel, {
    target,
    props: { state: new DetectionsSummaryState() },
  });
  await datasetsAvailability.init();
  await vi.waitFor(
    () => {
      flushSync();
      expect(
        target.querySelector(
          '[data-testid="detections-total"], [data-testid="detections-error"]',
        ),
      ).not.toBeNull();
    },
    { timeout: 5000 },
  );
}

const q = (id: string) => document.querySelector<HTMLElement>(`[data-testid="${id}"]`);

beforeEach(() => datasetsAvailability.reset());
afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
  datasetsAvailability.reset();
});

describe('DetectionsSummaryPanel', () => {
  it('renders the served totals, embedding chips, per-label table and truncation note', async () => {
    serve(SUMMARY);
    await render();
    expect(q('detections-total')!.textContent).toContain('120 detections');
    expect(target.textContent).toContain('embedded 100');
    expect(target.textContent).toContain('No vector: encoder failed 5');
    expect(target.textContent).toContain('Not embedded 15');
    expect(q('detections-by-label')!.textContent).toContain('widget');
    expect(q('detections-truncated')).not.toBeNull();
  });

  it('offers "Embed N detections" only with a served suggested_reprocess, and sends it as served', async () => {
    serve(SUMMARY);
    await render();
    const open = q('reprocess-open')!;
    expect(open.textContent?.trim()).toBe('Embed 20 detections');
    open.click();
    flushSync();
    [...document.querySelectorAll('button')]
      .find((b) => b.textContent?.trim() === 'Check what would run')!
      .click();
    await vi.waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0]!.body).toEqual(REQUEST);
  });

  it('shows no Embed action when suggested_reprocess is null', async () => {
    serve({ ...SUMMARY, suggested_reprocess: null });
    await render();
    expect(q('reprocess-open')).toBeNull();
  });

  it('shows the served error on a failed read', async () => {
    serve(null);
    await render();
    expect(q('detections-error')!.textContent).toContain('index down');
  });
});
