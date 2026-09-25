import { describe, expect, it } from 'vitest';
import { cohortCountKey, isAllZeroLoaded, splitCohortGroups } from './trainCohortGroups';

interface G {
  classId: number;
  cohorts: Array<{ id: string }>;
}

const cohorts = [{ id: 'validated' }, { id: 'needs_labeling' }];

function group(classId: number): G {
  return { classId, cohorts };
}

describe('cohortCountKey', () => {
  it('joins class id and cohort id with a colon', () => {
    expect(cohortCountKey(7, 'validated')).toBe('7:validated');
  });
});

describe('isAllZeroLoaded', () => {
  it('is false when cohort definitions have not loaded yet (empty cohorts array)', () => {
    const g: G = { classId: 1, cohorts: [] };
    expect(isAllZeroLoaded(g, {})).toBe(false);
  });

  it('is false when any count is still loading (undefined — not yet fetched)', () => {
    const g = group(1);
    const counts = { '1:validated': 0 }; // needs_labeling never fetched
    expect(isAllZeroLoaded(g, counts)).toBe(false);
  });

  it('is false when any count is null (a fetch failure, per loadGroupCounts)', () => {
    const g = group(1);
    const counts = { '1:validated': 0, '1:needs_labeling': null };
    expect(isAllZeroLoaded(g, counts)).toBe(false);
  });

  it('is false when any count is a nonzero served value', () => {
    const g = group(1);
    const counts = { '1:validated': 0, '1:needs_labeling': 16 };
    expect(isAllZeroLoaded(g, counts)).toBe(false);
  });

  it('is true only when every cohort served exactly 0', () => {
    const g = group(1);
    const counts = { '1:validated': 0, '1:needs_labeling': 0 };
    expect(isAllZeroLoaded(g, counts)).toBe(true);
  });
});

describe('splitCohortGroups', () => {
  it('keeps a not-yet-loaded class (empty cohorts) in the visible list, never the zero list', () => {
    const groups: G[] = [{ classId: 1, cohorts: [] }];
    const { visible, zero } = splitCohortGroups(groups, {});
    expect(visible).toEqual(groups);
    expect(zero).toEqual([]);
  });

  it('keeps a class with a still-loading count in the visible list', () => {
    const groups: G[] = [group(1)];
    const counts = { '1:validated': 0 }; // needs_labeling not in the map yet
    const { visible, zero } = splitCohortGroups(groups, counts);
    expect(visible).toEqual(groups);
    expect(zero).toEqual([]);
  });

  it('moves a class into the zero list once every cohort served 0', () => {
    const loaded = group(1);
    const stillLoading = group(2);
    const counts = {
      '1:validated': 0,
      '1:needs_labeling': 0,
      '2:validated': 0,
      // '2:needs_labeling' absent — still loading
    };
    const { visible, zero } = splitCohortGroups([loaded, stillLoading], counts);
    expect(visible).toEqual([stillLoading]);
    expect(zero).toEqual([loaded]);
  });

  it('preserves the original order within each bucket', () => {
    const a = group(1);
    const b = group(2);
    const c = group(3);
    const counts = {
      '1:validated': 0,
      '1:needs_labeling': 0,
      '2:validated': 5,
      '2:needs_labeling': 0,
      '3:validated': 0,
      '3:needs_labeling': 0,
    };
    const { visible, zero } = splitCohortGroups([a, b, c], counts);
    expect(visible).toEqual([b]);
    expect(zero).toEqual([a, c]);
  });
});

describe('splitCohortGroups pending (visual audit T4)', () => {
  it('counts groups whose definitions or counts are not all served yet', () => {
    const groups: G[] = [group(1), group(2), { classId: 3, cohorts: [] }];
    const counts = {
      '1:validated': 0,
      '1:needs_labeling': 0,
      '2:validated': 4,
    };
    const split = splitCohortGroups(groups, counts);
    expect(split.zero.map((g) => g.classId)).toEqual([1]);
    expect(split.pending).toBe(2);
  });

  it('is 0 once every group has every count served', () => {
    const counts = { '1:validated': 0, '1:needs_labeling': 3 };
    expect(splitCohortGroups([group(1)], counts).pending).toBe(0);
  });
});
