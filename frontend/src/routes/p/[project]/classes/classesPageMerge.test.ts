import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

/**
 * W4 (docs/design/logic-moves-adoption-plan-2026-09-24.md §2 W4) — the
 * merge dialog must call `POST {API_PREFIX}/classes/merge?dry_run=true`
 * before every real merge and disable Confirm when the server reports
 * `blocked`, and the class-name/hotkey rules must no longer be
 * re-implemented client-side. Static source-scan (no
 * @testing-library/svelte in this repo — see exportPageCleanup.test.ts).
 */
const here = path.dirname(fileURLToPath(import.meta.url));
const classesSrc = readFileSync(path.join(here, '+page.svelte'), 'utf-8');
const addClassModalSrc = readFileSync(
  path.resolve(here, '..', '..', '..', '..', 'lib', 'components', 'AddClassModal.svelte'),
  'utf-8',
);

describe('/classes: merge dry-run preview gates the real merge', () => {
  it('calls previewClassMerge (dry_run=true) rather than merging blind', () => {
    expect(classesSrc).toMatch(/previewClassMerge\(/);
  });

  it('disables the Confirm/Merge button when the dry-run reports blocked', () => {
    // The Merge button's `disabled` expression must reference
    // `mergePreview?.blocked` — this is what stops a real merge from
    // firing into a 409 the operator already knew was coming.
    const disabledBlock = classesSrc.slice(
      classesSrc.indexOf('void submitMerge()'),
      classesSrc.indexOf('void submitMerge()') + 400,
    );
    expect(disabledBlock).toMatch(/mergePreview\?\.blocked/);
  });

  it('submitMerge itself also refuses a blocked merge (defense in depth vs. a stale disabled state)', () => {
    const fn = classesSrc.slice(
      classesSrc.indexOf('async function submitMerge'),
      classesSrc.indexOf('async function syncToOpensearch'),
    );
    expect(fn).toMatch(/mergePreview\?\.blocked/);
  });
});

describe('/classes + AddClassModal: no client-side class-name slug regex', () => {
  it('the classes page no longer defines SLUG_RE/isSlug', () => {
    expect(classesSrc).not.toMatch(/SLUG_RE/);
    expect(classesSrc).not.toMatch(/function isSlug/);
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
