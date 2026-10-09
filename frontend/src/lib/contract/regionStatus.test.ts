/**
 * Region lifecycle-status contract: every region slot's `lifecycle`
 * capability must speak the backend's actual `RegionStatus` vocabulary
 * (`contracts/ts/regionStatus.ts`, vendored verbatim from
 * OpenProcessor's `src/config/region_state.py` via `npm run
 * contract:sync`) — not a hand-copied enum that can silently drift from
 * it.
 */
import { describe, expect, it } from 'vitest';
import { REGION_STATUS_VALUES, HUMAN_REGION_STATUSES } from '$contracts/ts/regionStatus';
import { regionContractSlots } from '$lib/test/fixtures/exampleProfiles';

const regionSlots = regionContractSlots();

describe('vendored region-status snapshot sanity', () => {
  it('loaded a non-trivial status set (guards a vacuous pass)', () => {
    expect(REGION_STATUS_VALUES.length).toBeGreaterThan(0);
    expect(HUMAN_REGION_STATUSES.length).toBeGreaterThan(0);
  });
});

describe('region slot lifecycle (served synthesis + region example profiles) vs the backend RegionStatus contract', () => {
  const lifecycleSlots = regionSlots.filter((s) => s.capabilities.lifecycle);

  it('at least one slot declares a lifecycle (guards a vacuous pass)', () => {
    expect(lifecycleSlots.length).toBeGreaterThan(0);
  });

  for (const slot of lifecycleSlots) {
    const lifecycle = slot.capabilities.lifecycle!;

    describe(slot.key, () => {
      it('declares exactly the backend RegionStatus values, no more no less', () => {
        const declared = lifecycle.states.map((s) => s.value).sort();
        expect(declared).toEqual([...REGION_STATUS_VALUES].sort());
      });

      it('flags exactly the backend HUMAN_REGION_STATUSES as human-writable', () => {
        const humanWritable = lifecycle.states
          .filter((s) => s.humanWritable)
          .map((s) => s.value)
          .sort();
        expect(humanWritable).toEqual([...HUMAN_REGION_STATUSES].sort());
      });

      it('confirmState is a human-writable status', () => {
        expect(HUMAN_REGION_STATUSES).toContain(lifecycle.confirmState);
      });

      it('rejectState is a human-writable status', () => {
        expect(HUMAN_REGION_STATUSES).toContain(lifecycle.rejectState);
      });

      it('falsePositiveState, if declared, is a human-writable status', () => {
        if (lifecycle.falsePositiveState) {
          expect(HUMAN_REGION_STATUSES).toContain(lifecycle.falsePositiveState);
        }
      });
    });
  }
});
