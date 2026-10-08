import { describe, it, expect } from 'vitest';
import { resolveSlotRegistry } from './registry';
import { readSlot } from './readSlot';
import { slotIsPresent } from './types';
import { loadExampleProfile } from '$lib/test/fixtures/exampleProfiles';
import { aircraftTailNumberSlot } from '$lib/test/fixtures/aircraftTailNumberSlot';
import { defectCodeSlot } from '$lib/test/fixtures/defectCodeSlot';
import { cohortsForClass, derivedCohorts } from './cohorts';
import type { XYXY } from './types';

// The license-plate example (examples/annotation-profiles/, never
// bundled), parsed the way a deployment's tier-2 file would be.
const licensePlateSlot = loadExampleProfile('license-plate.json').slots[0]!;

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

  it('a profile declaring lifecycle.states[].aliases resolves both the new and legacy raw value to the same state, with zero code outside profiles/', () => {
    const tail = registry.byKey('aircraft_tail_number')!;
    const renamed = {
      ...tail,
      capabilities: {
        ...tail.capabilities,
        lifecycle: {
          ...tail.capabilities.lifecycle!,
          states: tail.capabilities.lifecycle!.states.map((s) =>
            s.value === 'not_visible'
              ? { ...s, value: 'tail_not_visible', aliases: ['not_visible'] }
              : s,
          ),
        },
      },
    };
    const parent: XYXY = [0, 0, 1, 1];
    const viaNewValue = readSlot({ tail_status: 'tail_not_visible' }, renamed, parent);
    const viaLegacyAlias = readSlot({ tail_status: 'not_visible' }, renamed, parent);
    expect(viaNewValue.lifecycle?.state).toEqual(viaLegacyAlias.lifecycle?.state);
    expect(viaLegacyAlias.lifecycle?.state?.value).toBe('tail_not_visible');
  });

  it('none of the three slots share bind classes (no accidental collision)', () => {
    const classNames = [licensePlateSlot, aircraftTailNumberSlot, defectCodeSlot].map(
      (s) => s.bind.className,
    );
    expect(new Set(classNames).size).toBe(3);
  });
});

/**
 * P3.5 (docs/genericization-plan-2026-09-13.md §9.7/§9.9): extends the
 * falsification test to the cohort layer, added by the §9.2 addendum.
 * Derivation is pure and runs against the same three profiles — no
 * per-slot special-casing in cohorts.ts, registry.ts, or this file. If
 * either example needed a code change outside `profiles/` or
 * `licensePlateSlot`'s own capability declarations, that would be the
 * signal the cohort layer is wrong, exactly like the capability model
 * itself.
 */
describe('P3.5 — cohort-layer falsification: derivation generalizes with zero per-slot code', () => {
  it('aircraft_tail_number derives exactly blind_spots + low_conf + disagreement, never false_positives (no falsePositiveState)', () => {
    const ids = derivedCohorts(aircraftTailNumberSlot)
      .map((c) => c.id)
      .sort();
    expect(ids).toEqual(['blind_spots', 'disagreement', 'low_conf'].sort());
    expect(ids).not.toContain('false_positives');
  });

  it('defect_code derives no geometry cohort at all (no subBox) and no disagreement cohort (no chainField)', () => {
    const ids = derivedCohorts(defectCodeSlot).map((c) => c.id);
    expect(ids).toEqual([]);
  });

  it('a text-only slot (defect_code) is never silently required to have a box: cohortsForClass returns core cohorts only', () => {
    const { registry } = resolveSlotRegistry({ builtins: [defectCodeSlot] });
    const classesById = new Map([[11, 'part_surface']]);
    const cohorts = cohortsForClass(11, 'part_surface', registry, classesById, true);
    expect(cohorts.every((c) => c.rowKind !== 'slot')).toBe(true);
  });

  it("licensePlateSlot's declared cohorts replace the derived ids of the same name; the derived-only ids (blind_spots/low_conf) never leak through alongside them", () => {
    const { registry } = resolveSlotRegistry({ builtins: [licensePlateSlot] });
    const classesById = new Map([[3, 'license_plate']]);
    const cohorts = cohortsForClass(3, 'license_plate', registry, classesById, true);
    const ids = cohorts.map((c) => c.id);
    expect(ids).toContain('disagreement');
    expect(ids).toContain('false_positives');
    expect(ids).not.toContain('blind_spots');
    expect(ids).not.toContain('low_conf');
    // The declared 'disagreement'/'false_positives' are tier-1 endpoint
    // calls, not the derived tier-2 predicate — "declared is the ceiling."
    expect(cohorts.find((c) => c.id === 'disagreement')!.query.kind).toBe('endpoint');
    expect(cohorts.find((c) => c.id === 'false_positives')!.query.kind).toBe('endpoint');
  });
});
