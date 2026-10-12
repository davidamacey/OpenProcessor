import { describe, expect, it } from 'vitest';
import { progressFixture, runFixture } from '$lib/test/fixtures/bulkIngest';
import {
  formatDuration,
  headlineRate,
  progressPercent,
  stageRows,
} from './bulkIngestFormat';

describe('bulk ingest formatting', () => {
  it('formats durations and the unknown estimate', () => {
    expect(formatDuration(null)).toBe('--');
    expect(formatDuration(undefined)).toBe('--');
    expect(formatDuration(12.4)).toBe('12s');
    expect(formatDuration(249)).toBe('4m09s');
    expect(formatDuration(3900)).toBe('1h05m');
  });

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
});
