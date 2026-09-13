import { describe, it, expect } from 'vitest';
import { resolveSlotRegistry } from './registry';
import { readSlot } from './readSlot';
import { slotIsPresent } from './types';
import { licensePlateSlot } from './profiles/licensePlate';
import { aircraftTailNumberSlot } from './profiles/aircraftTailNumber';
import { defectCodeSlot } from './profiles/defectCode';
import type { XYXY } from './types';

/**
 * Phase 3 falsification test (docs/genericization-plan-2026-09-13.md
 * §5.4): loads all three profiles through the same registry/adapter
 * used by license_plate, and asserts the derived surface for each
 * without any per-slot special-casing in this file or in
 * registry.ts/readSlot.ts/types.ts. If either example needed a code
 * change outside `profiles/`, that would be the signal the capability
 * model is wrong — see the ship-gate note in the final report.
 */
describe('capability-model falsification: three independently-configured slots', () => {
  const { registry } = resolveSlotRegistry({
    builtins: [licensePlateSlot, aircraftTailNumberSlot, defectCodeSlot],
  });

  it('registers three distinct queue tabs with independent url/endpoint ids', () => {
    expect(registry.queues.map((s) => s.key)).toEqual([
      'license_plate',
      'aircraft_tail_number',
      'defect_code',
    ]);
    const tail = registry.byKey('aircraft_tail_number')!;
    expect(tail.capabilities.queue?.urlId).toBe('tails');
    expect(tail.capabilities.queue?.endpointId).toBe('tail_numbers');
  });

  it('aircraft_tail_number: no falsePositiveState means no markFalsePositive action key', () => {
    const tail = registry.byKey('aircraft_tail_number')!;
    expect(tail.capabilities.lifecycle?.falsePositiveState).toBeUndefined();
    expect(tail.capabilities.queue?.keymap.markFalsePositive).toBeUndefined();
    // 'f' must not appear anywhere in its keymap.
    const letters = Object.values(tail.capabilities.queue?.keymap ?? {}).flat();
    expect(letters).not.toContain('f');
  });

  it('aircraft_tail_number: a TALL envelope is honored, not the plate WIDE one', () => {
    const tail = registry.byKey('aircraft_tail_number')!;
    // storedFrame is 'parent' for this slot, so the box below is already
    // expressed as parent-relative fractions: w=0.1, h=0.4 -> aspect 0.25,
    // within [0.15, 1.4]; a plate's [1.2, 8.0] envelope would reject this
    // outright, which is exactly the leak this test is designed to catch.
    const parent: XYXY = [0, 0, 1, 1];
    const raw = { tail_bbox_norm: [0.4, 0.1, 0.5, 0.5] };
    const d = readSlot(raw, tail, parent);
    expect(d.subBox?.shapeWarning).toBe(false);
  });

  it('aircraft_tail_number: storedFrame=parent is read, not assumed source', () => {
    const tail = registry.byKey('aircraft_tail_number')!;
    const parent: XYXY = [0.2, 0.2, 0.6, 0.8];
    // Already parent-relative coordinates in [0,1].
    const raw = { tail_bbox_norm: [0.4, 0.1, 0.6, 0.5] };
    const d = readSlot(raw, tail, parent);
    expect(d.subBox?.parent?.cx).toBeCloseTo(0.5);
    expect(d.subBox?.parent?.cy).toBeCloseTo(0.3);
    expect(d.subBox?.parent?.w).toBeCloseTo(0.2);
    expect(d.subBox?.parent?.h).toBeCloseTo(0.4);
  });

  it('defect_code: has no subBox capability at all — geometry is structurally absent', () => {
    const defect = registry.byKey('defect_code')!;
    expect(defect.capabilities.subBox).toBeUndefined();
    const parent: XYXY = [0, 0, 1, 1];
    const d = readSlot(
      { defect_code: 'scratch', defect_status: 'confirmed' },
      defect,
      parent,
    );
    expect(d.subBox).toBeUndefined();
    expect(slotIsPresent(d)).toBe(true);
    expect(d.text?.value).toBe('scratch');
  });

  it('defect_code: text capability carries a closed vocabulary', () => {
    const defect = registry.byKey('defect_code')!;
    expect(defect.capabilities.text?.vocabulary?.map((v) => v.value)).toEqual([
      'none',
      'scratch',
      'dent',
      'corrosion',
      'weld',
    ]);
  });

  it('defect_code: no edit-box action is offered (no editBox key in its keymap)', () => {
    const defect = registry.byKey('defect_code')!;
    expect(defect.capabilities.queue?.keymap.editBox).toBeUndefined();
  });

  it('none of the three slots share bind classes (no accidental collision)', () => {
    const classNames = [licensePlateSlot, aircraftTailNumberSlot, defectCodeSlot].map(
      (s) => s.bind.className,
    );
    expect(new Set(classNames).size).toBe(3);
  });
});
