/**
 * Deprecate/Restore (OpenProcessor 01324cb, 243f7f2) — the `/classes`
 * Restore button used to be permanently `disabled` ("no backend support
 * exists"); that's now obsolete. `deprecateClassAction` must handle the
 * structured `class_still_referenced` 409 by offering the existing merge
 * flow (preselecting the class as merge source), and `restoreClassAction`
 * must surface restore's PLAIN STRING 409 detail verbatim — distinct
 * failure shapes, so each gets its own assertion.
 *
 * Static source-scan, matching this directory's existing convention
 * (classesPageMerge.test.ts, newClassProposals.test.ts) — no
 * @testing-library/svelte / component-mount harness for a full route page
 * in this repo.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

function fn(name: string, nextMarker: string): string {
  const start = src.indexOf(`async function ${name}(`);
  expect(start, `${name} not found`).toBeGreaterThan(-1);
  const end = src.indexOf(nextMarker, start);
  expect(end, `${nextMarker} not found after ${name}`).toBeGreaterThan(start);
  return src.slice(start, end);
}

describe('/classes: Restore is no longer permanently disabled', () => {
  it('no longer ships the disabled placeholder button', () => {
    expect(src).not.toMatch(/Restore not yet implemented/);
  });

  it("the deprecated table's Restore button calls restoreClassAction and is gated on busy, not disabled outright", () => {
    const idx = src.indexOf('data-testid="restore-{cls.id}"');
    expect(idx).toBeGreaterThan(-1);
    const block = src.slice(idx, idx + 200);
    expect(block).toMatch(/onclick=\{\(\) => void restoreClassAction\(cls\)\}/);
    expect(block).toMatch(/disabled=\{busy\}/);
  });
});

describe('/classes: Deprecate action per live class row', () => {
  it('renders a Deprecate button next to Rename on each active row', () => {
    const idx = src.indexOf('data-testid="deprecate-{cls.id}"');
    expect(idx).toBeGreaterThan(-1);
    const block = src.slice(idx, idx + 200);
    expect(block).toMatch(/onclick=\{\(\) => void deprecateClassAction\(cls\)\}/);
  });

  it('deprecateClassAction confirms naming the class before calling deprecateClass', () => {
    const f = fn('deprecateClassAction', 'async function restoreClassAction');
    expect(f).toMatch(/window\.confirm\(/);
    expect(f).toMatch(/cls\.name/);
    expect(f).toMatch(/await deprecateClass\(cls\.id\)/);
  });

  it('on success, toasts and refreshes classesStore so pickers/hotkeys update', () => {
    const f = fn('deprecateClassAction', 'async function restoreClassAction');
    expect(f).toMatch(/toastStore\.success\(/);
    expect(f).toMatch(/await classesStore\.clearAndRefetch\(\)/);
  });

  it('on a structured class_still_referenced 409, offers to open the merge flow with this class preselected as source', () => {
    const f = fn('deprecateClassAction', 'async function restoreClassAction');
    expect(f).toMatch(/classStillReferencedDetail\(e\)/);
    expect(f).toMatch(/ref\.item_count/);
    expect(f).toMatch(/ref\.confirmed_label_count/);
    expect(f).toMatch(/ref\.message/);
    expect(f).toMatch(/window\.confirm\(/);
    expect(f).toMatch(/openMergeWithSource\(cls\.id\)/);
  });

  it('a non-structured failure still falls back to a generic error toast', () => {
    const f = fn('deprecateClassAction', 'async function restoreClassAction');
    expect(f).toMatch(/toastStore\.error\(`Deprecate failed: \$\{apiErrorText\(e\)\}`\)/);
  });

  it('openMergeWithSource resets the merge dialog then pins mergeSourceId', () => {
    const idx = src.indexOf('function openMergeWithSource(');
    expect(idx).toBeGreaterThan(-1);
    const block = src.slice(idx, idx + 150);
    expect(block).toMatch(/openMerge\(\)/);
    expect(block).toMatch(/mergeSourceId = sourceId/);
  });
});

describe('/classes: Restore action', () => {
  it('restoreClassAction confirms naming the class before calling restoreClass', () => {
    const f = fn('restoreClassAction', 'function openAdd');
    expect(f).toMatch(/window\.confirm\(/);
    expect(f).toMatch(/cls\.name/);
    expect(f).toMatch(/await restoreClass\(cls\.id\)/);
  });

  it('on success, toasts and refreshes classesStore', () => {
    const f = fn('restoreClassAction', 'function openAdd');
    expect(f).toMatch(/toastStore\.success\(/);
    expect(f).toMatch(/await classesStore\.clearAndRefetch\(\)/);
  });

  it("shows the server's plain-string 409 detail verbatim, never the structured deprecate shape", () => {
    const f = fn('restoreClassAction', 'function openAdd');
    expect(f).toMatch(/e instanceof ApiError && e\.status === 409 && e\.detail/);
    expect(f).toMatch(/toastStore\.error\(e\.detail\)/);
    expect(f).not.toMatch(/classStillReferencedDetail/);
  });
});
