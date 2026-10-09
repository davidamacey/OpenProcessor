/**
 * Pure split for `/train`'s "Training cohorts" section (m-train-cohorts,
 * 2026-09-24 interactive pass): with ~85 classes × up to 8 cohorts, most
 * classes' chips are all 0 — a wall of zeros dominating the section
 * above the Past runs table. A class collapses into a "no candidates"
 * disclosure only once every one of its cohorts has SERVED a count of
 * exactly 0 — never `== null` (still loading, or a fetch failure — both
 * map to `null` in `+page.svelte`'s `loadGroupCounts`), so a class whose
 * counts haven't loaded yet (lazy IntersectionObserver hasn't fired, or
 * this session simply hasn't scrolled to it) always stays in the normal
 * list until the server actually says zero.
 */

export interface CohortGroupLike {
  classId: number;
  cohorts: Array<{ id: string }>;
}

/** Stable key across (class, cohort) pairs — mirrors `+page.svelte`'s
 *  `cohortKey` (a cohort id alone isn't unique once multiple classes
 *  share it, e.g. every class has a 'validated' core cohort). */
export function cohortCountKey(classId: number, cohortId: string): string {
  return `${classId}:${cohortId}`;
}

export function isAllZeroLoaded(
  group: CohortGroupLike,
  counts: Record<string, number | null | undefined>,
): boolean {
  if (group.cohorts.length === 0) return false; // cohort defs not loaded yet
  return group.cohorts.every((c) => counts[cohortCountKey(group.classId, c.id)] === 0);
}

export interface CohortGroupSplit<G extends CohortGroupLike> {
  visible: G[];
  zero: G[];
  /** T4 (visual audit 2026-09-24): groups whose cohort definitions or
   *  counts haven't all been served yet. While > 0 the "N classes with no
   *  candidates" count is provisional and says so. */
  pending: number;
}

export function splitCohortGroups<G extends CohortGroupLike>(
  groups: G[],
  counts: Record<string, number | null | undefined>,
): CohortGroupSplit<G> {
  const visible: G[] = [];
  const zero: G[] = [];
  let pending = 0;
  for (const g of groups) {
    (isAllZeroLoaded(g, counts) ? zero : visible).push(g);
    const loaded =
      g.cohorts.length > 0 &&
      g.cohorts.every((c) => typeof counts[cohortCountKey(g.classId, c.id)] === 'number');
    if (!loaded) pending += 1;
  }
  return { visible, zero, pending };
}
