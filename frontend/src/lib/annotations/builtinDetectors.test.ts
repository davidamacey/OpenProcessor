import { describe, it, expect } from 'vitest';
import { builtinDetectorRegistry } from './profiles/builtinDetectors';

/**
 * `builtinDetectorRegistry` now holds only the muted-tag outcome config
 * (labels/colors moved to the served
 * `GET {API_PREFIX}/regions/vocabulary` and `paletteForRole`, see
 * `detectorRegistry.test.ts` and `ProvenanceChip.test.ts`).
 */
describe('builtinDetectorRegistry (mutedTagPattern only)', () => {
  it('mutes miss/reject/skipped/degenerate/unparseable tags', () => {
    for (const tag of [
      'miss',
      'reject',
      'skipped',
      'degenerate',
      'unparseable',
      'gemma_reject',
      // `accepted_unverified` chain step —
      // muted, same as a miss/reject, so it reads as lower-confidence.
      'accepted_unverified',
    ]) {
      expect(builtinDetectorRegistry.mutedTagPattern.test(tag)).toBe(true);
    }
    expect(builtinDetectorRegistry.mutedTagPattern.test('hit')).toBe(false);
  });
});
