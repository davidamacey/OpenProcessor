import { describe, expect, it } from 'vitest';
import {
  CORE_COHORTS,
  cohortsForClass,
  compileCohortQuery,
  derivedCohorts,
} from './cohorts';
import { resolveSlotRegistry } from './registry';
import { licensePlateSlot } from './profiles/licensePlate';
import { aircraftTailNumberSlot } from './profiles/aircraftTailNumber';
import { defectCodeSlot } from './profiles/defectCode';

describe('compileCohortQuery', () => {
  it('substitutes {classId} in an endpoint query params bag', () => {
    const compiled = compileCohortQuery(
      { kind: 'endpoint', path: '/crops', params: { class_id: '{classId}' } },
      { classId: 42, className: 'sedan' },
    );
    expect(compiled).toEqual({
      kind: 'endpoint',
      path: '/crops',
      params: { class_id: '42' },
    });
  });

  it('leaves non-template values (booleans/numbers) untouched', () => {
    const compiled = compileCohortQuery(
      {
        kind: 'endpoint',
        path: '/crops',
        params: { label_validated: true, v6_conf_lt: 0.5 },
      },
      { classId: 1, className: 'x' },
    );
    expect(compiled).toEqual({
      kind: 'endpoint',
      path: '/crops',
      params: { label_validated: true, v6_conf_lt: 0.5 },
    });
  });

  it('passes tier-2 predicate queries through unchanged (no templates in that arm)', () => {
    const query: Parameters<typeof compileCohortQuery>[0] = {
      kind: 'predicate',
      filters: [{ field: 'plate_score', op: 'lt', value: 0.6 }],
      excludeTestHoldout: true,
    };
    expect(compileCohortQuery(query, { classId: 1, className: 'x' })).toBe(query);
  });
});

describe('cohortsForClass — class with no registered slot', () => {
  const { registry } = resolveSlotRegistry({ builtins: [licensePlateSlot] });
  const classesById = new Map([[7, 'sedan']]);

  it('gets exactly the 4 core cohorts, no slot cohorts', () => {
    const cohorts = cohortsForClass(7, 'sedan', registry, classesById, true);
    expect(cohorts.map((c) => c.id).sort()).toEqual(CORE_COHORTS.map((c) => c.id).sort());
  });

  it("compiles each core cohort's class_id against the requested class", () => {
    const cohorts = cohortsForClass(7, 'sedan', registry, classesById, true);
    const validated = cohorts.find((c) => c.id === 'validated')!;
    expect(validated.query).toEqual({
      kind: 'endpoint',
      path: '/crops',
      params: { class_id: '7', label_validated: true },
    });
  });
});

describe('cohortsForClass — license_plate (declared cohorts override derived)', () => {
  const { registry } = resolveSlotRegistry({ builtins: [licensePlateSlot] });
  const classesById = new Map([[3, 'license_plate']]);

  it('has 4 core + 5 declared LPR cohorts, not the weaker derived versions', () => {
    const cohorts = cohortsForClass(3, 'license_plate', registry, classesById, true);
    const ids = cohorts.map((c) => c.id).sort();
    expect(ids).toEqual(
      [
        ...CORE_COHORTS.map((c) => c.id),
        'lpr_blind_spots',
        'lpr_low_conf_correct',
        'disagreement',
        'human_corrected',
        'false_positives',
      ].sort(),
    );
  });

  it('every compiled tier-1 LPR cohort URL is byte-identical to getTrainingCandidates(mode, {class_id})', () => {
    const cohorts = cohortsForClass(3, 'license_plate', registry, classesById, true);
    for (const mode of [
      'lpr_blind_spots',
      'lpr_low_conf_correct',
      'disagreement',
      'human_corrected',
      'false_positives',
    ]) {
      const cohort = cohorts.find((c) => c.id === mode)!;
      expect(cohort.query).toEqual({
        kind: 'endpoint',
        path: '/plates/training_candidates',
        params: { mode, class_id: '3' },
      });
    }
  });

  it("declared 'disagreement' and 'false_positives' replace the derived ids of the same name", () => {
    const cohorts = cohortsForClass(3, 'license_plate', registry, classesById, true);
    const disagreement = cohorts.find((c) => c.id === 'disagreement')!;
    // The derived version would hit a predicate query on
    // plate_detector_chain — the declared one hits the real endpoint.
    expect(disagreement.query.kind).toBe('endpoint');
    const falsePositives = cohorts.find((c) => c.id === 'false_positives')!;
    expect(falsePositives.query.kind).toBe('endpoint');
  });
});

describe('derivedCohorts — falsification-adjacent (extended by P3.5)', () => {
  it('aircraft_tail_number (subBox + scoreField + chainField, no falsePositiveState) derives blind_spots/low_conf/disagreement, not false_positives', () => {
    const ids = derivedCohorts(aircraftTailNumberSlot).map((c) => c.id);
    expect(ids.sort()).toEqual(['blind_spots', 'disagreement', 'low_conf'].sort());
  });

  it('defect_code (no subBox, no chainField) derives no geometry cohort', () => {
    const ids = derivedCohorts(defectCodeSlot).map((c) => c.id);
    expect(ids).toEqual([]);
  });
});

describe('cohortsForClass — predicateCohortsAvailable gate', () => {
  it('drops every tier-2 derived cohort when the flag is off, keeping only core + declared', () => {
    const { registry } = resolveSlotRegistry({ builtins: [aircraftTailNumberSlot] });
    const classesById = new Map([[9, 'aircraft']]);
    const cohorts = cohortsForClass(9, 'aircraft', registry, classesById, false);
    expect(cohorts.map((c) => c.id).sort()).toEqual(CORE_COHORTS.map((c) => c.id).sort());
  });
});
