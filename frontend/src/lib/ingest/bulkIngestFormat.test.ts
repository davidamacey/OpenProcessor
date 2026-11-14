import { describe, expect, it } from 'vitest';
import { progressFixture, runFixture } from '$lib/test/fixtures/bulkIngest';
import {
  decodeRouteRows,
  headlineRate,
  progressPercent,
  stageRows,
} from './bulkIngestFormat';

describe('bulk ingest formatting', () => {
  it('bounds the bar and has none before anything is found', () => {
    expect(progressPercent(progressFixture())).toBe(0);
    expect(progressPercent(progressFixture({ discovered: 200, processed: 50 }))).toBe(25);
    expect(progressPercent(progressFixture({ discovered: 10, processed: 99 }))).toBe(100);
  });

  it('headlines the recent rate only while live', () => {
    const p = progressFixture({ images_per_s: 20, images_per_s_recent: 25 });
    expect(headlineRate(p, true)).toBe(25);
    expect(headlineRate(p, false)).toBe(20);
    expect(headlineRate(progressFixture({ images_per_s: 20 }), true)).toBe(20);
    expect(headlineRate(progressFixture(), true)).toBeNull();
  });

  it('lists stages largest first', () => {
    const run = runFixture({ stage_s: { crop: 1, embed: 9, decode: 3 } });
    expect(stageRows(run).map(([name]) => name)).toEqual(['embed', 'decode', 'crop']);
  });

  it('lists decode routes largest first and nothing when none are served', () => {
    const run = runFixture({ decode_routes: { cpu_small: 2, gpu: 90, cpu_corrupt: 5 } });
    expect(decodeRouteRows(run)).toEqual([
      ['gpu', 90],
      ['cpu_corrupt', 5],
      ['cpu_small', 2],
    ]);
    expect(decodeRouteRows(runFixture())).toEqual([]);
  });
});
