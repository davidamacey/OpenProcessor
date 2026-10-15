/**
 * The training-cohort preview used to render once, after every class group in
 * the `{#each cohortGroups}` loop (now `visibleCohortGroups`, see
 * `$lib/trainCohortGroups.ts`): clicking a chip near the top of the list
 * opened the preview thousands of pixels below, off-screen, which looked like
 * the click had done nothing. It renders inline under the group whose chip
 * was clicked.
 *
 * Static source scan (no component-mount harness for this page).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(
  path.resolve(here, '../../../../lib/components/train/TrainingCohortsPanel.svelte'),
  'utf-8',
);
const controller = readFileSync(
  path.resolve(here, '../../../../lib/train/trainCohortsController.svelte.ts'),
  'utf-8',
);

describe('the cohort preview renders inline under the class group whose chip was clicked', () => {
  it('the preview {#if selectedCohortKey} block lives inside the {#each cohortGroups} block, not after it', () => {
    const groupsBlock = src.match(
      /\{#each ctl\.visibleCohortGroups as group \(group\.classId\)\}[\s\S]*?\n {2}\{\/each\}/,
    )?.[0];
    expect(groupsBlock).not.toBeUndefined();
    expect(groupsBlock).toMatch(/\{#if ctl\.selectedCohortKey\}/);
    expect(groupsBlock).toMatch(/ctl\.cohortPreviewError/);
  });

  it("the active cohort is looked up within this group's own cohorts, not a flatMap over every group", () => {
    const groupsBlock = src.match(
      /\{#each ctl\.visibleCohortGroups as group \(group\.classId\)\}[\s\S]*?\n {2}\{\/each\}/,
    )?.[0];
    expect(groupsBlock).toMatch(
      /group\.cohorts\.find\(\s*\(c\) => ctl\.cohortKey\(group\.classId, c\) === ctl\.selectedCohortKey,?\s*\)/,
    );
    for (const s of [src, controller]) {
      expect(s).not.toMatch(/cohortGroups\s*\n?\s*\.flatMap\(\(g\) => g\.cohorts\.map/);
    }
  });
});
