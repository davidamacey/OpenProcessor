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
import { datasetStatsFixture } from '$lib/test/fixtures/datasetStats';
import {
  installServedRegionProfile,
  resetDeploymentSlots,
} from '$lib/annotations/registeredSlots';
import { WIDGET_TAG_PROFILE } from '$lib/test/fixtures/regionSlot';

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

type StatsOverrides = {
  [K in keyof DatasetStatsType]?: DatasetStatsType[K] extends object
    ? Partial<DatasetStatsType[K]>
    : DatasetStatsType[K];
};

/** A full served `/stats/dataset` body; nested sections merge over it. */
function goodStats(overrides: StatsOverrides = {}): DatasetStatsType {
  const base = datasetStatsFixture();
  return {
    ...base,
    ...overrides,
    labeled: { ...base.labeled, ...overrides.labeled },
    regions: { ...base.regions, ...overrides.regions },
    unlabeled: { ...base.unlabeled, ...overrides.unlabeled },
    in_progress: { ...base.in_progress, ...overrides.in_progress },
    clusters: { ...base.clusters, ...overrides.clusters },
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
  resetDeploymentSlots();
});

describe('DatasetStats', () => {
  it('renders the labeled total as the real sum of all label sources', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(DatasetStats, { target, props: {} });
    flushSync();

    capturedOpts?.onSnapshot?.({}, goodStats() as unknown as Record<string, unknown>);
    flushSync();

    // 100 + 200 + 50 + 5 = 355
    const totalLabel = Array.from(target.querySelectorAll('span')).find((s) =>
      s.textContent?.includes('355 total'),
    );
    expect(totalLabel).toBeTruthy();
  });

  it('shows the current cluster total and, separately, what the last run made', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(DatasetStats, { target, props: {} });
    flushSync();

    capturedOpts?.onSnapshot?.(
      {},
      goodStats({
        clusters: {
          last_run_at: null,
          cluster_count: 106,
          last_run_cluster_count: 1,
          residual_count: 0,
          noise_count: 0,
          method: null,
        },
      }) as unknown as Record<string, unknown>,
    );
    flushSync();

    const rows = Array.from(target.querySelectorAll('dl div')).map((d) =>
      d.textContent?.replace(/\s+/g, ' ').trim(),
    );
    expect(rows).toContain('Clusters (total now) 106');
    expect(rows).toContain('Made by last run 1');
  });

  it('#36 item 2 (D1): renders unlabeled.vlm_no_class in the Unlabeled block', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(DatasetStats, { target, props: {} });
    flushSync();

    capturedOpts?.onSnapshot?.(
      {},
      goodStats({
        unlabeled: {
          pending_detection: 10,
          pending_verification: 5,
          no_label_source: 1252,
          vlm_no_class: 1252,
        },
      }) as unknown as Record<string, unknown>,
    );
    flushSync();

    const rows = Array.from(target.querySelectorAll('dl div')).map((d) =>
      d.textContent?.replace(/\s+/g, ' ').trim(),
    );
    expect(rows).toContain('VLM, no class 1,252');
  });

  it('F-69: the Unlabeled header shows the served class-less count, never a sum of overlapping buckets', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(DatasetStats, { target, props: {} });
    flushSync();

    // Pending detection overlaps the class-less bucket: a sum (3,516 +
    // 539) would exceed total_crops (3,516).
    capturedOpts?.onSnapshot?.(
      {},
      goodStats({
        total_crops: 3516,
        unlabeled: {
          pending_detection: 3516,
          pending_verification: 0,
          no_label_source: 539,
        },
      }) as unknown as Record<string, unknown>,
    );
    flushSync();

    const header = target.querySelector('[data-testid="unlabeled-header-count"]');
    expect(header?.textContent?.trim()).toBe('539 without a class');
    expect(target.textContent).not.toContain('4,055');
  });

  it('F-23: renders unlabeled.by_proposal, and no labeled "Proposal" row', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(DatasetStats, { target, props: {} });
    flushSync();

    capturedOpts?.onSnapshot?.(
      {},
      goodStats({
        unlabeled: {
          pending_detection: 0,
          pending_verification: 0,
          no_label_source: 40,
          by_proposal: 17,
        },
      }) as unknown as Record<string, unknown>,
    );
    flushSync();

    const rows = Array.from(target.querySelectorAll('dl div')).map((d) =>
      d.textContent?.replace(/\s+/g, ' ').trim(),
    );
    expect(rows).toContain('Detector proposal, no class 17');
    expect(target.textContent).not.toContain('Proposal (unclassified)');
  });

  it('V-1: renders the served region_stall_reason verbatim, and nothing when null', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(DatasetStats, { target, props: {} });
    flushSync();

    const reason = 'segmenter unavailable since 2026-09-25T10:00:00Z (model not ready)';
    capturedOpts?.onSnapshot?.(
      {},
      goodStats({
        in_progress: { region_drain_total_unfinished: 12, region_stall_reason: reason },
      }) as unknown as Record<string, unknown>,
    );
    flushSync();
    expect(
      target.querySelector('[data-testid="region-stall-reason"]')?.textContent,
    ).toContain(reason);

    capturedOpts?.onStats?.(
      goodStats({
        in_progress: { region_drain_total_unfinished: 12, region_stall_reason: null },
      }) as unknown as Record<string, unknown>,
    );
    flushSync();
    expect(target.querySelector('[data-testid="region-stall-reason"]')).toBeNull();
  });

  it('D4 (visual audit 2026-09-24): no hardcoded model/vendor names or HDD copy, verifier count named as such', () => {
    installServedRegionProfile(WIDGET_TAG_PROFILE);
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(DatasetStats, { target, props: {} });
    flushSync();

    capturedOpts?.onSnapshot?.(
      {},
      goodStats({
        regions: {
          total_detected: 30,
          boxed: 30,
          confirmed: 30,
          by_detector: 20,
          by_segmenter: 10,
        },
        in_progress: { region_drain_total_unfinished: 3 },
      } as StatsOverrides) as unknown as Record<string, unknown>,
    );
    flushSync();

    const text = target.textContent ?? '';
    for (const stale of ['Gemma', 'SAM', 'HDD']) {
      expect(text).not.toContain(stale);
    }
    expect(text).toContain('Distinct sources');
    expect(text).toContain('Pending detection');
    expect(text).toContain('verifier-confirmed');
    // The panel is titled by the served profile's display name.
    expect(text).toContain(WIDGET_TAG_PROFILE.display_name);
  });

  it('no region profile: no detections panel at all (domain-neutral audit §5.4)', () => {
    installServedRegionProfile(null);
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(DatasetStats, { target, props: {} });
    flushSync();

    capturedOpts?.onSnapshot?.(
      {},
      goodStats({
        regions: {
          total_detected: 30,
          boxed: 30,
          confirmed: 30,
          by_detector: 20,
          by_segmenter: 10,
        },
      } as StatsOverrides) as unknown as Record<string, unknown>,
    );
    flushSync();

    const text = target.textContent ?? '';
    expect(text).toContain('Distinct sources');
    expect(text).not.toContain('verifier-confirmed');
    expect(text).not.toContain('No detections yet');
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

  it('shows the served embedding counts and per-state chips, "unknown" verbatim', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(DatasetStats, { target, props: {} });
    flushSync();

    capturedOpts?.onSnapshot?.(
      {},
      goodStats({
        embedding: {
          embedded: 900,
          not_embedded: 100,
          by_state: {
            embedded: 900,
            not_selected: 60,
            deferred: 0,
            failed: 15,
            unknown: 25,
          },
        },
      }) as unknown as Record<string, unknown>,
    );
    flushSync();

    const card = target.querySelector('[data-testid="dataset-embedding"]')!;
    expect(card.textContent).toContain('900');
    expect(card.textContent).toContain('100');
    const chips = [
      ...card.querySelectorAll('[data-testid="embedding-by-state"] span'),
    ].map((c) => c.textContent?.replace(/\s+/g, ' ').trim());
    // Only states with a count are chips; legacy items report as `unknown`.
    expect(chips).toEqual([
      'No vector: encoder failed 15',
      'Not embedded 60',
      'unknown 25',
    ]);
  });

  it('renders no Embedding card (and no error) when the served stats carry none', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(DatasetStats, { target, props: {} });
    flushSync();

    const stats = goodStats() as unknown as Record<string, unknown>;
    delete stats.embedding;
    capturedOpts?.onSnapshot?.({}, stats);
    flushSync();

    expect(target.querySelector('[data-testid="dataset-embedding"]')).toBeNull();
    expect(target.textContent).toContain('Clusters (total now)');
  });
});
