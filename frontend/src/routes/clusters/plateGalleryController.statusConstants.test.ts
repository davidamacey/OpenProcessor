/**
 * C4b (docs/design/slot-generic-crop-mapping-plan-2026-09-21.md §7.3):
 * plateGalleryController.svelte.ts and SlotGallery.svelte used to
 * hardcode 'false_positive' | 'no_plate_visible' | 'detected' as raw
 * lifecycle-state literals — a state literal that silently means
 * nothing for another slot, and (per the plan) the highest-risk class
 * of straggler in this codebase. Both files now read
 * PLATE_CONFIRM_STATE / PLATE_REJECT_STATE / PLATE_FALSE_POSITIVE_STATE,
 * derived from licensePlateSlot itself, so a backend rename only ever
 * needs editing licensePlate.ts.
 *
 * This is the "zero direct test coverage" controller (own header,
 * §7.3 of the plan) — full behavioral coverage is P2.7/F7's job, not
 * this commit's. This test's only job is to pin that the exported
 * constants stay derived from (never drift from) the profile.
 */
import { describe, expect, it } from 'vitest';
import {
  PLATE_CONFIRM_STATE,
  PLATE_REJECT_STATE,
  PLATE_FALSE_POSITIVE_STATE,
} from './plateGalleryController.svelte';
import { licensePlateSlot } from '$lib/annotations/profiles/licensePlate';

describe('plateGalleryController status constants', () => {
  // Kept (test-audit-2026-09-24.md T2/P2-2): this is a real regression
  // guard — it fails if PLATE_CONFIRM_STATE/etc. are ever re-hardcoded
  // instead of derived from the profile. No mount/controller test
  // reaches this: it's a module-level `export const` computed once at
  // import time, not something a component render or a controller call
  // exercises.
  it('are read from licensePlateSlot, not hardcoded', () => {
    const lifecycle = licensePlateSlot.capabilities.lifecycle!;
    expect(PLATE_CONFIRM_STATE).toBe(lifecycle.confirmState);
    expect(PLATE_REJECT_STATE).toBe(lifecycle.rejectState);
    expect(PLATE_FALSE_POSITIVE_STATE).toBe(lifecycle.falsePositiveState);
  });

  // Deleted (test-audit-2026-09-24.md T2): "today's values match the
  // pre-C4b hardcoded literals" only re-asserted the profile's own
  // literal values back at itself — it never fails for a real
  // regression, only when someone edits licensePlate.ts's lifecycle
  // states on purpose, and the test would get updated in the same
  // commit.
});
