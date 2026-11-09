/*
 * Live dataset-stats feed for the pipeline-stats panel: subscribes to the
 * pipeline SSE stream and keeps the last-known stats, the transport state
 * and the drain-rate samples. Call during component initialisation (it
 * registers a $effect).
 *   * `snapshot` on connect carries the initial dataset stats.
 *   * `stats` is pushed when the rollup actually changed.
 *   * `state` frames belong to the pipeline-status panel and are ignored.
 */
import type { DatasetStats } from '$lib/api_stats';
import { subscribePipelineEvents } from '$lib/sse';
import {
  appendDrainSample,
  drainRateOf,
  etaSecondsOf,
  resolveStatsUpdate,
} from '$lib/datasetStats';
import type { DrainSample } from '$lib/datasetStats';

export function createDatasetStatsFeed() {
  let stats = $state<DatasetStats | null>(null);
  let error = $state<string | null>(null);
  let lastUpdated = $state<number | null>(null);
  let connected = $state<boolean>(false);
  let samples = $state<DrainSample[]>([]);

  function applyStats(payload: Record<string, unknown>): void {
    // The stats body can be an `{error}` envelope; resolveStatsUpdate keeps
    // the last-known-good stats in that case.
    const result = resolveStatsUpdate(payload, stats);
    stats = result.stats;
    error = result.error;
    if (result.error) return;
    const now = Date.now();
    lastUpdated = now;
    samples = appendDrainSample(
      samples,
      stats?.in_progress?.region_drain_total_unfinished ?? 0,
      now,
    );
  }

  $effect(() => {
    const subscription = subscribePipelineEvents({
      onSnapshot: (_state, statsPayload) => {
        connected = true;
        applyStats(statsPayload);
      },
      onStats: (statsPayload) => {
        connected = true;
        applyStats(statsPayload);
      },
      // Keep the last-known stats on screen while the browser reconnects.
      onError: () => {
        connected = false;
      },
      onOpen: () => {
        connected = true;
      },
    });
    return () => subscription.close();
  });

  const drainRate = $derived(drainRateOf(samples));
  const etaSeconds = $derived(
    etaSecondsOf(drainRate, stats?.in_progress?.region_drain_total_unfinished ?? 0),
  );

  return {
    get stats() {
      return stats;
    },
    get error() {
      return error;
    },
    get lastUpdated() {
      return lastUpdated;
    },
    get connected() {
      return connected;
    },
    get drainRate() {
      return drainRate;
    },
    get etaSeconds() {
      return etaSeconds;
    },
  };
}
