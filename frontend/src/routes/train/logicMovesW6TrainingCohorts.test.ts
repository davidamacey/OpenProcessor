/**
 * W6 training cohorts (docs/design/logic-moves-adoption-plan-2026-09-24.md
 * item 13) — /train's cohort picker sources its definitions from
 * `GET {API_PREFIX}/training_cohorts?class_id=`, not the client-only
 * `cohortsForClass()` (CORE_COHORTS + hardcoded licensePlateSlot modes).
 * This repo has no component-mount harness (see
 * clusters/[id]/clusterMoveRace.test.ts's doc comment), so this is a
 * static source scan, same convention as datasetExportGate.test.ts.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.resolve(here, './+page.svelte'), 'utf-8');

describe('W6: /train cohort definitions come from GET /training_cohorts?class_id=', () => {
  it('imports getTrainingCohorts and the ServedTrainingCohort type from $lib/api', () => {
    expect(src).toMatch(/getTrainingCohorts/);
    expect(src).toMatch(/ServedTrainingCohort/);
  });

  it('loadGroupCohorts calls getTrainingCohorts(group.classId) and maps served cohorts, not cohortsForClass alone', () => {
    const fn = src.match(/async function loadGroupCohorts\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toBeDefined();
    expect(fn).toMatch(/getTrainingCohorts\(group\.classId\)/);
    expect(fn).toMatch(/servedCohorts/);
  });

  it('a tier-2 declared cohort only fills in an id the server did not already send', () => {
    const fn = src.match(/async function loadGroupCohorts\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toMatch(/!servedIds\.has\(c\.id\)/);
  });

  it('cohort definitions load lazily per group, not eagerly for every class on mount', () => {
    // cohortGroups must read from a per-class map (classCohorts), never
    // call cohortsForClass directly to populate every group up front.
    const groupsBlock = src.match(
      /const cohortGroups = \$derived\.by<CohortGroup\[\]>\(([\s\S]*?)\n {2}\);/,
    )?.[0];
    expect(groupsBlock).toBeDefined();
    expect(groupsBlock).not.toMatch(/cohortsForClass\(/);
    expect(groupsBlock).toMatch(/classCohorts\[c\.id\]/);
  });

  it('runCohortQuery forwards params.classifier_conf_lt/label_validated verbatim rather than a client constant', () => {
    const fn = src.match(/async function runCohortQuery\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toBeDefined();
    expect(fn).toMatch(/params\.classifier_conf_lt/);
    expect(fn).toMatch(/params\.label_validated/);
  });
});
