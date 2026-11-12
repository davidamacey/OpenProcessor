/**
 * Region bbox edit session (coordinator finding 2026-09-25). The real
 * behavior proof is e2e/stubbed/test_region_bbox_edit_nudge.py; these
 * source checks pin the three wiring rules a jsdom mount of this route
 * can't reach (the page isn't mountable in this harness).
 */
import { describe, expect, it } from 'vitest';
import { reviewSource } from '$lib/testing/reviewPageSources';
import { extractFunction, normalize } from '$lib/testing/sourceScan';

const editSrc = normalize(reviewSource('lib/review/reviewSlotEditController.svelte.ts'));
const actionsSrc = normalize(reviewSource('lib/review/reviewSlotActions.svelte.ts'));
const keyboardSrc = normalize(reviewSource('lib/review/reviewKeyboard.svelte.ts'));

describe('/review region bbox edit session', () => {
  it('the reseed effect depends only on the crop id and runs its body untracked', () => {
    expect(editSrc).toContain(
      '$effect(() => { const id = current?.id; untrack(() => reseedForCrop(id ?? null)); });',
    );
  });

  it('Enter in edit mode saves the boxes of the crop it captured, not a re-read of `current` after the await', () => {
    const fn = extractFunction(actionsSrc, 'confirmMultiBoxSlot');
    expect(fn).toContain('const item = current;');
    expect(fn).toContain('multiBox.confirmAndSave(item.id)');
    expect(fn).not.toContain('confirmAndSave(current');
  });

  it('N / Z are not registered while editing, so the queue cannot move under an edit', () => {
    // K1 (configurable-keyboard-shortcuts plan): registrations are by
    // keymap action id; `review.skip`/`review.undo` default to n/z.
    expect(keyboardSrc).toContain(
      "if (!editMode) { reg('review.skip', skip); reg('review.undo', undoLast); }",
    );
  });
});
