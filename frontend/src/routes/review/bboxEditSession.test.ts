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

  it('Enter in edit mode saves to the crop the session started on', () => {
    const fn = extractFunction(src, 'saveBboxAndExit');
    expect(fn).toContain('const id = editingCropId ?? current.id;');
    expect(fn).toContain('if (id !== current.id)');
  });

  it('N / Z are not registered while editing, so the queue cannot move under an edit', () => {
    expect(src).toContain(
      "if (!editMode) { reg('n', skip, 'Skip'); reg('z', undoLast, 'Undo last'); }",
    );
  });
});
