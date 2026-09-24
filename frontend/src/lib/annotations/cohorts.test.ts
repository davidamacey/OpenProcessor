import { describe, expect, it } from 'vitest';
import {
  CORE_COHORTS,
  cohortEndpointKind,
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
      filters: [{ field: 'region_score', op: 'lt', value: 0.6 }],
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
        'detector_blind_spots',
        'low_conf_correct',
        'disagreement',
        'human_corrected',
        'false_positives',
      ].sort(),
    );
  });

  // Wave 2 C13/C14 (docs/design/slot-generic-crop-mapping-plan-2026-09-21.md
  // §8.4(iii)): the cohort `id` (render key) was split from `params.mode`
  // (the wire value) in C13; C14 flips `mode` and `path` to match the
  // backend's live rename (/plates -> /regions, lpr_* -> the new names).
  it('every compiled tier-1 LPR cohort URL is byte-identical to getTrainingCandidates(mode, {class_id})', () => {
    const cohorts = cohortsForClass(3, 'license_plate', registry, classesById, true);
    const idToMode: Record<string, string> = {
      detector_blind_spots: 'detector_blind_spots',
      low_conf_correct: 'low_conf_correct',
      disagreement: 'disagreement',
      human_corrected: 'human_corrected',
      false_positives: 'false_positives',
    };
    for (const [id, mode] of Object.entries(idToMode)) {
      const cohort = cohorts.find((c) => c.id === id)!;
      expect(cohort.query).toEqual({
        kind: 'endpoint',
        path: '/regions/training_candidates',
        params: { mode, class_id: '3' },
      });
    }
  });

  it("declared 'disagreement' and 'false_positives' replace the derived ids of the same name", () => {
    const cohorts = cohortsForClass(3, 'license_plate', registry, classesById, true);
    const disagreement = cohorts.find((c) => c.id === 'disagreement')!;
    // The derived version would hit a predicate query on
    // region_detector_chain — the declared one hits the real endpoint.
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

describe('cohortEndpointKind (Wave 2 C12 — structural dispatch, not path === literal)', () => {
  it('recognizes training_candidates under the old /plates base', () => {
    expect(cohortEndpointKind('/plates/training_candidates')).toBe('training_candidates');
  });

  it('recognizes training_candidates under the renamed /regions base — the whole point of the fix', () => {
    expect(cohortEndpointKind('/regions/training_candidates')).toBe(
      'training_candidates',
    );
  });

  it("recognizes a hypothetical second slot's differently-based endpoint cohort", () => {
    expect(cohortEndpointKind('/widgets/training_candidates')).toBe(
      'training_candidates',
    );
  });

  it('recognizes /crops and /review/model_disagreements regardless of base', () => {
    expect(cohortEndpointKind('/crops')).toBe('crops');
    expect(cohortEndpointKind('/review/model_disagreements')).toBe('model_disagreements');
  });

  it('fails closed (null) for an unrecognized endpoint shape', () => {
    expect(cohortEndpointKind('/regions/suspected_false_positives')).toBeNull();
  });
});
