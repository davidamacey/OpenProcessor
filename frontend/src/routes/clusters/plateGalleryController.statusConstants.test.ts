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
  it('are read from licensePlateSlot, not hardcoded', () => {
    const lifecycle = licensePlateSlot.capabilities.lifecycle!;
    expect(PLATE_CONFIRM_STATE).toBe(lifecycle.confirmState);
    expect(PLATE_REJECT_STATE).toBe(lifecycle.rejectState);
    expect(PLATE_FALSE_POSITIVE_STATE).toBe(lifecycle.falsePositiveState);
  });

  it("today's values match the pre-C4b hardcoded literals (no behavior change)", () => {
    expect(PLATE_CONFIRM_STATE).toBe('detected');
    expect(PLATE_REJECT_STATE).toBe('no_plate_visible');
    expect(PLATE_FALSE_POSITIVE_STATE).toBe('false_positive');
  });
});
