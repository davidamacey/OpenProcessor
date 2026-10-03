/**
 * Piece 11 (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §A.6):
 * the server-path batch panel, gated on served `batch.source_roots`
 * being non-empty (the page only mounts this component in that case —
 * see `src/routes/ingest/+page.svelte` — so this test doesn't re-cover
 * that gate, only the panel's own submit/render behavior).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import IngestBatchPanel from './IngestBatchPanel.svelte';
import { ingestBatch } from '$lib/api';
import { resolveIngestConfig } from '$lib/ingest/ingestConfig';
import { servedIngestConfig } from '$lib/test/fixtures/ingestConfig';
import type { BatchIngestResponse } from '$lib/types';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, ingestBatch: vi.fn() };
});

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target.remove();
  vi.restoreAllMocks();
});

function servedResponse(
  overrides: Partial<BatchIngestResponse> = {},
): BatchIngestResponse {
  return {
    status: 'partial',
    summary: {
      successful: 1,
      duplicates: 0,
      failed: 1,
      crops_indexed: 1,
      n_embedded: 0,
      n_not_embedded: 0,
      n_embed_failed: 0,
      n_filtered: 0,
    },
    results: [
      {
        status: 'success',
        image_id: 'img-1',
        image_path: '/data/a.jpg',
        imohash: 'h',
        n_crops: 1,
        n_regions: 0,
        n_embedded: 0,
        n_not_embedded: 0,
        n_embed_failed: 0,
        n_filtered: 0,
        error: null,
        error_kind: null,
        source_identifier: null,
      },
      {
        status: 'failed',
        image_id: '',
        image_path: '/data/b.jpg',
        imohash: '',
        n_crops: 0,
        n_regions: 0,
        n_embedded: 0,
        n_not_embedded: 0,
        n_embed_failed: 0,
        n_filtered: 0,
        error: 'not a servable path',
        error_kind: 'unservable_path',
        source_identifier: null,
      },
    ],
    ...overrides,
  };
}

function textarea(): HTMLTextAreaElement {
  return target.querySelector('textarea')!;
}

describe('IngestBatchPanel', () => {
  it('renders the served source roots read-only', () => {
    instance = mount(IngestBatchPanel, {
      target,
      props: {
        config: {
          ...resolveIngestConfig(servedIngestConfig()),
          batchSourceRoots: ['/data/archive', '/data/incoming'],
        },
      },
    });
    flushSync();
    expect(target.textContent).toContain('/data/archive');
    expect(target.textContent).toContain('/data/incoming');
  });

  it('submits POST /ingest/batch with the entered paths and shows the summary', async () => {
    vi.mocked(ingestBatch).mockResolvedValue(servedResponse());
    instance = mount(IngestBatchPanel, {
      target,
      props: {
        config: {
          ...resolveIngestConfig(servedIngestConfig()),
          batchSourceRoots: ['/data/archive'],
        },
      },
    });
    flushSync();
    const ta = textarea();
    ta.value = '/data/archive/a.jpg\n/data/archive/b.jpg';
    ta.dispatchEvent(new Event('input'));
    flushSync();
    const submitBtn = [...target.querySelectorAll('button')].find((b) =>
      b.textContent?.includes('Ingest 2 paths'),
    );
    expect(submitBtn).toBeTruthy();
    submitBtn!.click();
    await vi.waitFor(() => {
      expect(ingestBatch).toHaveBeenCalled();
    });
    const call = vi.mocked(ingestBatch).mock.calls[0]![0];
    expect(call.items.map((i) => i.path)).toEqual([
      '/data/archive/a.jpg',
      '/data/archive/b.jpg',
    ]);
    // Only a path and a source tag per item, and nothing else on the
    // request: labeled data comes in through the dataset import.
    expect(call).toEqual({
      items: [
        { path: '/data/archive/a.jpg', source: 'batch' },
        { path: '/data/archive/b.jpg', source: 'batch' },
      ],
    });
    flushSync();
    await vi.waitFor(() => {
      flushSync();
      expect(target.textContent).toContain('successful 1');
    });
    expect(target.textContent).toContain('failed 1');
  });

  it('groups and filters failures by the served error_kind', async () => {
    vi.mocked(ingestBatch).mockResolvedValue(servedResponse());
    instance = mount(IngestBatchPanel, {
      target,
      props: {
        config: {
          ...resolveIngestConfig(servedIngestConfig()),
          batchSourceRoots: ['/data/archive'],
        },
      },
    });
    flushSync();
    textarea().value = '/data/archive/b.jpg';
    textarea().dispatchEvent(new Event('input'));
    flushSync();
    const submitBtn = [...target.querySelectorAll('button')].find((b) =>
      b.textContent?.includes('Ingest 1 path'),
    );
    submitBtn!.click();
    await vi.waitFor(() => {
      flushSync();
      expect(target.textContent).toContain('unservable_path');
    });
    expect(target.textContent).toContain('not a servable path');
  });

  it('offers no label-import controls (one paths box, one source tag)', () => {
    instance = mount(IngestBatchPanel, {
      target,
      props: {
        config: {
          ...resolveIngestConfig(servedIngestConfig()),
          batchSourceRoots: ['/data/archive'],
        },
      },
    });
    flushSync();
    expect(target.querySelectorAll('textarea')).toHaveLength(1);
    expect(target.querySelectorAll('input[type="checkbox"]')).toHaveLength(0);
    expect(target.textContent).not.toMatch(/label|mismatch/i);
  });

  it('blocks submit when the entered path count exceeds the served batch.max_items', () => {
    instance = mount(IngestBatchPanel, {
      target,
      props: {
        config: {
          ...resolveIngestConfig(servedIngestConfig()),
          batchSourceRoots: ['/data/archive'],
          batchMaxItems: 1,
        },
      },
    });
    flushSync();
    textarea().value = '/data/archive/a.jpg\n/data/archive/b.jpg';
    textarea().dispatchEvent(new Event('input'));
    flushSync();
    expect(target.textContent).toContain('exceeds');
    const submitBtn = [...target.querySelectorAll('button')].find((b) =>
      b.textContent?.includes('Ingest 2 paths'),
    );
    expect(submitBtn?.disabled).toBe(true);
  });

  it('d72cc63: lists served secondary-detector failures', async () => {
    const base = servedResponse();
    vi.mocked(ingestBatch).mockResolvedValue({
      ...base,
      summary: { ...base.summary, secondary_detector_failures: 1 },
      results: [
        { ...base.results[0]!, secondary_detector_error: 'classifier unavailable' },
        base.results[1]!,
      ],
    });
    instance = mount(IngestBatchPanel, {
      target,
      props: {
        config: {
          ...resolveIngestConfig(servedIngestConfig()),
          batchSourceRoots: ['/data/archive'],
        },
      },
    });
    flushSync();
    textarea().value = '/data/archive/a.jpg';
    textarea().dispatchEvent(new Event('input'));
    flushSync();
    [...target.querySelectorAll('button')]
      .find((b) => b.textContent?.includes('Ingest 1 path'))!
      .click();
    await vi.waitFor(() => {
      flushSync();
      expect(
        target.querySelector('[data-testid="batch-secondary-failures"]')?.textContent,
      ).toContain('secondary detector failed 1');
    });
    expect(
      target.querySelector('[data-testid="batch-secondary-list"]')?.textContent,
    ).toContain('classifier unavailable');
  });

  it('shows the served embedding totals and lists the files left without a vector', async () => {
    const base = servedResponse();
    vi.mocked(ingestBatch).mockResolvedValue({
      ...base,
      summary: {
        ...base.summary,
        n_embedded: 7,
        n_not_embedded: 2,
        n_embed_failed: 1,
        n_filtered: 3,
      },
      results: [
        { ...base.results[0]!, n_embedded: 5, n_not_embedded: 2, n_embed_failed: 1 },
      ],
    });
    instance = mount(IngestBatchPanel, {
      target,
      props: {
        config: {
          ...resolveIngestConfig(servedIngestConfig()),
          batchSourceRoots: ['/data/archive'],
        },
      },
    });
    flushSync();
    const ta = textarea();
    ta.value = '/data/archive/a.jpg';
    ta.dispatchEvent(new Event('input'));
    flushSync();
    [...target.querySelectorAll('button')]
      .find((b) => b.textContent?.includes('Ingest 1 path'))!
      .click();
    await vi.waitFor(() => {
      flushSync();
      expect(
        target.querySelector('[data-testid="batch-embedding-totals"]'),
      ).not.toBeNull();
    });
    const totals = target.querySelector(
      '[data-testid="batch-embedding-totals"]',
    )!.textContent!;
    expect(totals).toContain('embedded 7');
    expect(totals).toContain('not embedded 2');
    expect(totals).toContain('encoder failed 1');
    expect(totals).toContain('filtered out 3');
    expect(
      target.querySelector('[data-testid="batch-not-embedded-list"]')?.textContent,
    ).toContain('/data/a.jpg');
  });

  it('the submit button is disabled with no paths (never sends an empty items list)', () => {
    instance = mount(IngestBatchPanel, {
      target,
      props: {
        config: {
          ...resolveIngestConfig(servedIngestConfig()),
          batchSourceRoots: ['/data/archive'],
        },
      },
    });
    flushSync();
    const btn = [...target.querySelectorAll('button')].find((b) =>
      b.textContent?.includes('Ingest 0 paths'),
    );
    expect(btn?.disabled).toBe(true);
  });
});
