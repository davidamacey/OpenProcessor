/**
 * Mount-based behavior test for DatasetStats (docs/design/test-audit-2026-09-24.md
 * P1-4). Covers the exact regression the audit's mutation check found
 * surviving: dropping `l.by_vlm` from the labeled total (§2.2) — this is
 * that "an {error} payload shows Stats unavailable and keeps the last good
 * values" and "the labeled total is a real sum" behavior, asserted on the
 * rendered DOM rather than the source text.
 *
 * `subscribePipelineEvents` is mocked so the test can push `snapshot`/
 * `stats` frames directly instead of standing up a real EventSource.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import type { DatasetStats as DatasetStatsType } from '$lib/api';

type PipelineOpts = {
  onSnapshot?: (state: Record<string, unknown>, stats: Record<string, unknown>) => void;
  onStats?: (stats: Record<string, unknown>) => void;
  onError?: () => void;
  onOpen?: () => void;
};

let capturedOpts: PipelineOpts | null = null;

vi.mock('$lib/sse', () => ({
  subscribePipelineEvents: (opts: PipelineOpts) => {
    capturedOpts = opts;
    return { close: vi.fn() };
  },
}));

const { default: DatasetStats } = await import('./DatasetStats.svelte');

function goodStats(overrides: Partial<DatasetStatsType> = {}): DatasetStatsType {
  return {
    as_of: '2026-09-24T00:00:00Z',
    total_crops: 1000,
    validated: 500,
    test_holdout: 50,
    by_source: [{ key: 'nas1', doc_count: 1000 }],
    labeled: { by_human: 100, by_vlm: 200, by_classifier: 50, by_proposal: 10, other: 5 },
    regions: { total_detected: 0, by_detector: 0, by_segmenter: 0, by_human: 0 },
    unlabeled: { pending_detection: 10, pending_verification: 5, no_label_source: 2 },
    in_progress: { sam_drain_total_unfinished: 0 },
    clusters: {
      last_run_at: null,
      cluster_count: 0,
      residual_count: 0,
      noise_count: 0,
      method: null,
    },
    ...overrides,
  } as DatasetStatsType;
}

let target: HTMLDivElement;
let instance: unknown;

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
  capturedOpts = null;
});

describe('DatasetStats', () => {
  it('renders the labeled total as the real sum of all label sources', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(DatasetStats, { target, props: {} });
    flushSync();

    capturedOpts?.onSnapshot?.({}, goodStats() as unknown as Record<string, unknown>);
    flushSync();

    // 100 + 200 + 50 + 10 + 5 = 365
    const totalLabel = Array.from(target.querySelectorAll('span')).find((s) =>
      s.textContent?.includes('365 total'),
    );
    expect(totalLabel).toBeTruthy();
  });

  it('shows "Stats unavailable" on an {error} payload and keeps the last good values', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(DatasetStats, { target, props: {} });
    flushSync();

    capturedOpts?.onSnapshot?.(
      {},
      goodStats({ total_crops: 42_000 }) as unknown as Record<string, unknown>,
    );
    flushSync();
    expect(target.textContent).toContain('42,000');

    capturedOpts?.onStats?.({ error: 'op_items mapping unavailable' });
    flushSync();

    expect(target.textContent).toContain('Stats unavailable');
    expect(target.textContent).toContain('op_items mapping unavailable');
    // Last-known-good total is still on screen, not blanked.
    expect(target.textContent).toContain('42,000');
  });

  it('m29 (2026-09-24 interactive pass): drops the green "live" badge to "degraded" once a frame errors', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(DatasetStats, { target, props: {} });
    flushSync();

    capturedOpts?.onSnapshot?.({}, goodStats() as unknown as Record<string, unknown>);
    flushSync();
    const badgeBefore = Array.from(target.querySelectorAll('span')).find(
      (s) => s.textContent?.trim() === 'live',
    );
    expect(badgeBefore).toBeTruthy();
    expect(badgeBefore?.className).toContain('bg-emerald-500');

    capturedOpts?.onStats?.({ error: 'opensearch unavailable' });
    flushSync();

    const liveBadgeAfter = Array.from(target.querySelectorAll('span')).find(
      (s) => s.textContent?.trim() === 'live',
    );
    const degradedBadge = Array.from(target.querySelectorAll('span')).find(
      (s) => s.textContent?.trim() === 'degraded',
    );
    expect(liveBadgeAfter).toBeUndefined();
    expect(degradedBadge).toBeTruthy();
    expect(degradedBadge?.className).not.toContain('bg-emerald-500');
  });

  it('m29: truncates a long raw error in the banner text but keeps it available via title', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(DatasetStats, { target, props: {} });
    flushSync();

    const raw =
      "HTTPException: 503: opensearch error: RequestError(400, 'search_phase_execution_exception', 'Text fields are not optimised for operations that require per-document field data like aggregations and sorting')";
    capturedOpts?.onSnapshot?.({}, { error: raw });
    flushSync();

    expect(target.textContent).not.toContain(raw);
    const banner = Array.from(target.querySelectorAll('[title]')).find((el) =>
      el.textContent?.includes('Stats unavailable'),
    );
    expect(banner?.getAttribute('title')).toBe(raw);
  });
});
