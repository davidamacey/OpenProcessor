import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

/**
 * the
 * merge dialog must call `POST {API_PREFIX}/classes/merge?dry_run=true`
 * before every real merge and disable Confirm when the server reports
 * `blocked`, and the class-name/hotkey rules must no longer be
 * re-implemented client-side. Static source-scan (no
 * @testing-library/svelte in this repo — see exportPageCleanup.test.ts).
 */
const here = path.dirname(fileURLToPath(import.meta.url));
const LIB = path.resolve(here, '../../../../lib');
const classesSrc = readFileSync(path.join(here, '+page.svelte'), 'utf-8');
const mergeSrc = readFileSync(
  path.join(LIB, 'classes/mergeClassesController.svelte.ts'),
  'utf-8',
);
const mergeModalSrc = readFileSync(
  path.join(LIB, 'components/classes/MergeClassesModal.svelte'),
  'utf-8',
);
const addClassModalSrc = readFileSync(
  path.resolve(here, '..', '..', '..', '..', 'lib', 'components', 'AddClassModal.svelte'),
  'utf-8',
);

describe('/classes: merge dry-run preview gates the real merge', () => {
  it('calls previewClassMerge (dry_run=true) rather than merging blind', () => {
    expect(mergeSrc).toMatch(/previewClassMerge\(/);
  });

  it('disables the Confirm/Merge button when the dry-run reports blocked', () => {
    // The Merge button's `disabled` expression must reference
    // `mergePreview?.blocked` — this is what stops a real merge from
    // firing into a 409 the operator already knew was coming.
    const disabledBlock = mergeModalSrc.slice(
      mergeModalSrc.indexOf('void m.submitMerge()'),
      mergeModalSrc.indexOf('void m.submitMerge()') + 400,
    );
    expect(disabledBlock).toMatch(/mergePreview\?\.blocked/);
  });

  it('submitMerge itself also refuses a blocked merge (defense in depth vs. a stale disabled state)', () => {
    const fn = mergeSrc.slice(
      mergeSrc.indexOf('async function submitMerge'),
      mergeSrc.indexOf('\n  return {'),
    );
    expect(fn).toMatch(/mergePreview\?\.blocked/);
  });

  it('cannot be confirmed without a served dry run (no client-substituted count)', () => {
    const disabledBlock = mergeModalSrc.slice(
      mergeModalSrc.indexOf('void m.submitMerge()'),
      mergeModalSrc.indexOf('void m.submitMerge()') + 400,
    );
    expect(disabledBlock).toMatch(/mergePreview == null/);
    const fn = mergeSrc.slice(
      mergeSrc.indexOf('async function submitMerge'),
      mergeSrc.indexOf('\n  return {'),
    );
    expect(fn).toMatch(/if \(mergePreview == null\) return;/);
    expect(fn).not.toMatch(/validated_count/);
  });
});

describe('/classes + AddClassModal: no client-side class-name slug regex', () => {
  it('the classes page no longer defines SLUG_RE/isSlug', () => {
    for (const s of [classesSrc, mergeSrc]) {
      expect(s).not.toMatch(/SLUG_RE/);
      expect(s).not.toMatch(/function isSlug/);
    }
  });

  it('AddClassModal no longer defines or tests against SLUG_RE', () => {
    expect(addClassModalSrc).not.toMatch(/SLUG_RE/);
  });

  it('AddClassModal has no native pattern= duplicating the server slug rule', () => {
    expect(addClassModalSrc).not.toMatch(/pattern="\[a-z0-9_\]\+"/);
  });
});

describe('/classes: reserved-hotkey conflict banner (live widget_a→b, 2026-09-24)', () => {
  it('derives reservedConflicts from reservedHotkeyLetters() + bound classes', () => {
    expect(classesSrc).toMatch(/reservedHotkeyLetters\(\)/);
    expect(classesSrc).toMatch(/reservedConflicts/);
  });
});
