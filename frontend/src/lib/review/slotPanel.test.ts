import { describe, expect, it } from 'vitest';
import {
  humanWritableStates,
  statusClearsBox,
  statusWantsRejectionReason,
  panelLabels,
} from './slotPanel';
import { licensePlateSlot } from '../annotations/profiles/licensePlate';
import { aircraftTailNumberSlot } from '../annotations/profiles/aircraftTailNumber';

describe('humanWritableStates', () => {
  it('licensePlateSlot: matches the pre-generalization PLATE_STATUS_OPTIONS order/values', () => {
    expect(humanWritableStates(licensePlateSlot)).toEqual([
      { value: 'detected', label: 'detected (plate visible)' },
      { value: 'verify_rejected', label: 'rejected (bad detection)' },
      { value: 'no_region_visible', label: 'no plate visible' },
      { value: 'false_positive', label: 'false positive (keep box)' },
    ]);
  });

  it('aircraftTailNumberSlot: a different profile produces a different set with zero code change', () => {
    expect(humanWritableStates(aircraftTailNumberSlot)).toEqual([
      { value: 'detected', label: 'detected' },
      { value: 'not_visible', label: 'no tail number visible' },
      { value: 'obscured', label: 'obscured / partial' },
    ]);
  });
});

describe('statusClearsBox', () => {
  it('licensePlateSlot: only rejectState clears the box', () => {
    expect(statusClearsBox(licensePlateSlot, 'no_region_visible')).toBe(true);
    expect(statusClearsBox(licensePlateSlot, 'detected')).toBe(false);
    expect(statusClearsBox(licensePlateSlot, '')).toBe(false);
  });

  it('aircraftTailNumberSlot: its own rejectState (not_visible), not the plate one', () => {
    expect(statusClearsBox(aircraftTailNumberSlot, 'not_visible')).toBe(true);
    expect(statusClearsBox(aircraftTailNumberSlot, 'no_plate_visible')).toBe(false);
  });
});

describe('statusWantsRejectionReason', () => {
  it("licensePlateSlot: exactly {verify_rejected, no_region_visible} — the no-regression proof for the old '=== verify_rejected || === no_plate_visible' check", () => {
    const wants = licensePlateSlot.capabilities
      .lifecycle!.states.map((s) => s.value)
      .filter((v) => statusWantsRejectionReason(licensePlateSlot, v));
    expect(wants.sort()).toEqual(['no_region_visible', 'verify_rejected'].sort());
  });

  it('aircraftTailNumberSlot: obscured (rejected) and not_visible (absent), not detected (proposed)', () => {
    expect(statusWantsRejectionReason(aircraftTailNumberSlot, 'obscured')).toBe(true);
    expect(statusWantsRejectionReason(aircraftTailNumberSlot, 'not_visible')).toBe(true);
    expect(statusWantsRejectionReason(aircraftTailNumberSlot, 'detected')).toBe(false);
  });
});

describe('panelLabels', () => {
  it('licensePlateSlot: reproduces every literal string the pre-generalization panel hardcoded', () => {
    expect(panelLabels(licensePlateSlot)).toEqual({
      scoreLabel: 'Plate score',
      statusLabel: 'Plate status',
      textLabel: 'Plate text',
      textPlaceholder: 'ABC123',
      confirmLabel: 'Confirm Plate',
      rejectLabel: 'Reject (no plate)',
      noBoxHint: 'No plate bbox on this crop — press E to draw one.',
    });
  });

  it('aircraftTailNumberSlot: a completely different label set, zero code change', () => {
    expect(panelLabels(aircraftTailNumberSlot)).toEqual({
      scoreLabel: 'Tail number score',
      statusLabel: 'Tail number status',
      textLabel: 'Tail number',
      textPlaceholder: 'N123AB',
      confirmLabel: 'Confirm Tail number',
      rejectLabel: 'Reject (no tail number)',
      noBoxHint: 'No tail number bbox on this crop — press E to draw one.',
    });
  });
});
