/**
 * Regression test for M14 (docs/design/interactive-pass-2026-09-24.md
 * §6 FRONTEND): the training-cohort preview used to render once, after
 * every class group in the `{#each cohortGroups}` loop — clicking a
 * chip near the top of the list opened the preview thousands of pixels
 * below, off-screen, which looked like the click had done nothing.
 *
 * Same static source-scan convention as logicMovesW6TrainingCohorts.test.ts
 * (no component-mount harness for this page).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.resolve(here, './+page.svelte'), 'utf-8');

describe('M14: the cohort preview renders inline under the class group whose chip was clicked', () => {
  it('the preview {#if selectedCohortKey} block lives inside the {#each cohortGroups} block, not after it', () => {
    const groupsBlock = src.match(
      /\{#each cohortGroups as group \(group\.classId\)\}[\s\S]*?\n {4}\{\/each\}/,
    )?.[0];
    expect(groupsBlock).toBeDefined();
    expect(groupsBlock).toMatch(/\{#if selectedCohortKey\}/);
    expect(groupsBlock).toMatch(/cohortPreviewError/);
  });

  it("the active cohort is looked up within this group's own cohorts, not a flatMap over every group", () => {
    const groupsBlock = src.match(
      /\{#each cohortGroups as group \(group\.classId\)\}[\s\S]*?\n {4}\{\/each\}/,
    )?.[0];
    expect(groupsBlock).toMatch(
      /group\.cohorts\.find\(\s*\(c\) => cohortKey\(group\.classId, c\) === selectedCohortKey,?\s*\)/,
    );
    expect(src).not.toMatch(/cohortGroups\s*\n?\s*\.flatMap\(\(g\) => g\.cohorts\.map/);
  });
});
