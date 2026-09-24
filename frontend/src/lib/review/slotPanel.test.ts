import { describe, expect, it } from 'vitest';
import {
  humanWritableStates,
  statusClearsBox,
  statusWantsRejectionReason,
  panelLabels,
} from './slotPanel';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
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
  it("widgetTagSlot: the slot's own human-writable states, in declared order", () => {
    expect(humanWritableStates(widgetTagSlot)).toEqual([
      { value: 'detected', label: 'detected' },
      { value: 'verify_rejected', label: 'verify rejected' },
      { value: 'no_region_visible', label: 'no region visible' },
      { value: 'false_positive', label: 'false positive' },
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
    // SERVED's labels/order deliberately differ from widgetTagSlot's own
    // declared states, and omits false_positive — proves served wins.
    expect(humanWritableStates(widgetTagSlot, SERVED)).toEqual([
      { value: 'detected', label: 'detected (region visible)' },
      { value: 'verify_rejected', label: 'rejected (bad detection)' },
      { value: 'no_region_visible', label: 'no region visible' },
    ]);
  });

  it('falls back to the profile when served is empty', () => {
    expect(humanWritableStates(widgetTagSlot, [])).toEqual(
      humanWritableStates(widgetTagSlot),
    );
  });
});

describe('statusClearsBox', () => {
  it('widgetTagSlot: only rejectState clears the box', () => {
    expect(statusClearsBox(widgetTagSlot, 'no_region_visible')).toBe(true);
    expect(statusClearsBox(widgetTagSlot, 'detected')).toBe(false);
    expect(statusClearsBox(widgetTagSlot, '')).toBe(false);
  });

  it("aircraftTailNumberSlot: its own rejectState (not_visible), not another slot's", () => {
    expect(statusClearsBox(aircraftTailNumberSlot, 'not_visible')).toBe(true);
    expect(statusClearsBox(aircraftTailNumberSlot, 'no_region_visible')).toBe(false);
  });

  it('reads clears_box off the served vocabulary when present', () => {
    expect(statusClearsBox(widgetTagSlot, 'no_region_visible', SERVED)).toBe(true);
    expect(statusClearsBox(widgetTagSlot, 'detected', SERVED)).toBe(false);
    // Unknown-to-SERVED value -> false, not a throw.
    expect(statusClearsBox(widgetTagSlot, 'false_positive', SERVED)).toBe(false);
  });
});

describe('statusWantsRejectionReason', () => {
  it('widgetTagSlot: exactly {verify_rejected, no_region_visible} (the rejected + absent roles)', () => {
    const wants = widgetTagSlot.capabilities
      .lifecycle!.states.map((s) => s.value)
      .filter((v) => statusWantsRejectionReason(widgetTagSlot, v));
    expect(wants.sort()).toEqual(['no_region_visible', 'verify_rejected'].sort());
  });

  it('aircraftTailNumberSlot: obscured (rejected) and not_visible (absent), not detected (proposed)', () => {
    expect(statusWantsRejectionReason(aircraftTailNumberSlot, 'obscured')).toBe(true);
    expect(statusWantsRejectionReason(aircraftTailNumberSlot, 'not_visible')).toBe(true);
    expect(statusWantsRejectionReason(aircraftTailNumberSlot, 'detected')).toBe(false);
  });

  it('reads wants_reason off the served vocabulary when present', () => {
    expect(statusWantsRejectionReason(widgetTagSlot, 'verify_rejected', SERVED)).toBe(
      true,
    );
    expect(statusWantsRejectionReason(widgetTagSlot, 'no_region_visible', SERVED)).toBe(
      true,
    );
    expect(statusWantsRejectionReason(widgetTagSlot, 'detected', SERVED)).toBe(false);
    // human_writable: false on SERVED -> false even if wants_reason were true.
    expect(statusWantsRejectionReason(widgetTagSlot, 'pending_detection', SERVED)).toBe(
      false,
    );
  });
});

describe('panelLabels', () => {
  it("widgetTagSlot: every label comes from the slot's own label/text capability", () => {
    expect(panelLabels(widgetTagSlot)).toEqual({
      scoreLabel: 'Tag score',
      statusLabel: 'Tag status',
      textLabel: 'Tag text',
      textPlaceholder: 'TAG-001',
      confirmLabel: 'Confirm Tag',
      rejectLabel: 'Reject (no tag)',
      noBoxHint: 'No tag bbox on this crop — press E to draw one.',
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
