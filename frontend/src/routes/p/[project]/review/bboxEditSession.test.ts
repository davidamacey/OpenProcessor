/**
 * Region bbox edit session (coordinator finding 2026-09-25). The real
 * behavior proof is e2e/stubbed/test_region_bbox_edit_nudge.py; these
 * source checks pin the three wiring rules a jsdom mount of this route
 * can't reach (the page isn't mountable in this harness).
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { extractFunction, normalize } from '$lib/testing/sourceScan';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = normalize(readFileSync(path.join(here, '+page.svelte'), 'utf-8'));

describe('/review region bbox edit session', () => {
  it('the reseed effect depends only on the crop id and runs its body untracked', () => {
    expect(src).toContain(
      '$effect(() => { const id = current?.id; untrack(() => reseedForCrop(id ?? null)); });',
    );
  });

  it('Enter in edit mode saves the boxes of the crop it captured, not a re-read of `current` after the await', () => {
    const fn = extractFunction(src, 'confirmMultiBoxSlot');
    expect(fn).toContain('const item = current;');
    expect(fn).toContain('multiBox.confirmAndSave(item.id)');
    expect(fn).not.toContain('confirmAndSave(current');
  });

  it('N / Z are not registered while editing, so the queue cannot move under an edit', () => {
    // K1 (configurable-keyboard-shortcuts plan): registrations are by
    // keymap action id; `review.skip`/`review.undo` default to n/z.
    expect(src).toContain(
      "if (!editMode) { reg('review.skip', skip); reg('review.undo', undoLast); }",
    );
  });
});
