import { describe, expect, it } from 'vitest';
import {
  humanWritableStates,
  statusClearsBox,
  statusWantsRejectionReason,
  panelLabels,
} from './slotPanel';
import { licensePlateSlot } from '../annotations/profiles/licensePlate';
import { aircraftTailNumberSlot } from '../annotations/profiles/aircraftTailNumber';
import type { RegionStatusEntry } from '../api';

const SERVED: RegionStatusEntry[] = [
  {
    value: 'detected',
    label: 'detected (region visible)',
    role: 'positive',
    terminal: true,
    human_writable: true,
    clears_box: false,
    wants_reason: false,
  },
  {
    value: 'verify_rejected',
    label: 'rejected (bad detection)',
    role: 'rejected',
    terminal: true,
    human_writable: true,
    clears_box: false,
    wants_reason: true,
  },
  {
    value: 'no_region_visible',
    label: 'no region visible',
    role: 'absent',
    terminal: true,
    human_writable: true,
    clears_box: true,
    wants_reason: true,
  },
  {
    value: 'pending_detection',
    label: 'pending detection',
    role: 'pending',
    terminal: false,
    human_writable: false,
    clears_box: false,
    wants_reason: false,
  },
];

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

  it('prefers the served vocabulary over the profile fallback when present', () => {
    // SERVED's labels/order deliberately differ from licensePlateSlot's own
    // declared states, and omits false_positive — proves served wins.
    expect(humanWritableStates(licensePlateSlot, SERVED)).toEqual([
      { value: 'detected', label: 'detected (region visible)' },
      { value: 'verify_rejected', label: 'rejected (bad detection)' },
      { value: 'no_region_visible', label: 'no region visible' },
    ]);
  });

  it('falls back to the profile when served is empty', () => {
    expect(humanWritableStates(licensePlateSlot, [])).toEqual(
      humanWritableStates(licensePlateSlot),
    );
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

  it('reads clears_box off the served vocabulary when present', () => {
    expect(statusClearsBox(licensePlateSlot, 'no_region_visible', SERVED)).toBe(true);
    expect(statusClearsBox(licensePlateSlot, 'detected', SERVED)).toBe(false);
    // Unknown-to-SERVED value -> false, not a throw.
    expect(statusClearsBox(licensePlateSlot, 'false_positive', SERVED)).toBe(false);
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

  it('reads wants_reason off the served vocabulary when present', () => {
    expect(statusWantsRejectionReason(licensePlateSlot, 'verify_rejected', SERVED)).toBe(
      true,
    );
    expect(
      statusWantsRejectionReason(licensePlateSlot, 'no_region_visible', SERVED),
    ).toBe(true);
    expect(statusWantsRejectionReason(licensePlateSlot, 'detected', SERVED)).toBe(false);
    // human_writable: false on SERVED -> false even if wants_reason were true.
    expect(
      statusWantsRejectionReason(licensePlateSlot, 'pending_detection', SERVED),
    ).toBe(false);
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
