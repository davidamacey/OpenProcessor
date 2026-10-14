/**
 * Validator unit tests (docs/design/tier2-annotation-profile-config-plan-
 * 2026-09-20.md §5.2's 67 enumerated cases). Each invalid case asserts
 * `result.slot` is `undefined` AND that `errors` names the offending
 * field — a message that doesn't say what's wrong isn't a usable
 * operator-facing diagnostic.
 *
 * `any` is used, deliberately and only in this file, to mutate deeply
 * nested fixture objects into DELIBERATELY malformed shapes (wrong
 * types, missing required fields, injected forbidden keys) — the exact
 * opposite of what a fully-typed fixture could express. The parser under
 * test is `unknown`-typed throughout and never trusts these shapes.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

import { describe, expect, it } from 'vitest';
import { parseSlotConfig } from './parseSlotConfig';
import { expectRejected, fullSlot } from '$lib/test/fixtures/slotConfigInputs';

describe('parseSlotConfig — cohorts', () => {
  it('53. cohort query.path with {classId} accepted', () => {
    const s = fullSlot();
    const r = parseSlotConfig(s);
    expect(r.slot).toBeDefined();
    expect(r.slot!.capabilities.trainingCohorts!.cohorts[0].query).toMatchObject({
      kind: 'endpoint',
      path: '/widgets/training_candidates',
    });
  });

  it('54. cohort query.path using {cropId} rejected (wrong placeholder set)', () => {
    const s = fullSlot();
    (s.capabilities as any).trainingCohorts.cohorts[0].query.path =
      '/widgets/training_candidates/{cropId}';
    expectRejected(parseSlotConfig(s), /query\.path/);
  });

  it('55. cohort params key with a space rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).trainingCohorts.cohorts[0].query.params = { 'class id': 1 };
    expectRejected(parseSlotConfig(s), /params key/);
  });

  it('56. cohort params value with ";" rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).trainingCohorts.cohorts[0].query.params = { mode: 'a;b' };
    expectRejected(parseSlotConfig(s), /params\.mode/);
  });

  it('57. cohort query.kind: "sql" rejected', () => {
    const s = fullSlot();
    (s.capabilities as any).trainingCohorts.cohorts[0].query = { kind: 'sql' };
    expectRejected(parseSlotConfig(s), /query\.kind/);
  });
});
