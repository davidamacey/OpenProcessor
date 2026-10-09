/**
 * MultiBoxCanvas lock glyph (W10): a box whose served `locked` is true
 * carries a lock glyph; the rest do not. Presentation only.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { reprocessVocabularyStore } from '$lib/stores/reprocessVocabulary.svelte';
import { reprocessVocabularyFixture } from '$lib/test/fixtures/regionProfiles';
import MultiBoxCanvas, { type CanvasBox } from './MultiBoxCanvas.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  reprocessVocabularyStore.resetForProjectChange();
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

const box = (cx: number, over: Partial<CanvasBox> = {}): CanvasBox => ({
  box: { cx, cy: 0.5, w: 0.2, h: 0.2 },
  state: 'accepted',
  label: 'Tag (accepted)',
  ...over,
});

function render(boxes: CanvasBox[]) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(MultiBoxCanvas, {
    target,
    props: { cropId: 'c_1', boxes, selectedIndex: null, readonly: true },
  });
  flushSync();
  return target;
}

describe('MultiBoxCanvas lock glyph', () => {
  it('marks exactly the locked boxes', () => {
    reprocessVocabularyStore.vocabulary = reprocessVocabularyFixture();
    reprocessVocabularyStore.loaded = true;
    const el = render([
      box(0.2, { locked: true }),
      box(0.5),
      box(0.8, { locked: false }),
    ]);
    const rows = [...el.querySelectorAll('[data-box-index]')];
    expect(rows).toHaveLength(3);
    expect(rows[0]!.querySelector('[data-testid="box-locked"]')).not.toBeNull();
    expect(rows[1]!.querySelector('[data-testid="box-locked"]')).toBeNull();
    expect(rows[2]!.querySelector('[data-testid="box-locked"]')).toBeNull();
    expect(
      rows[0]!.querySelector('[data-testid="box-locked"]')?.getAttribute('title'),
    ).toBe(
      'Locked. Locked when: Human label: Human label: served description; Validated: Validated: served description; Imported: Imported: served description; Test holdout: Test holdout: served description',
    );
  });

  it('shows no lock glyph when no box is locked', () => {
    const el = render([box(0.3), box(0.7)]);
    expect(el.querySelector('[data-testid="box-locked"]')).toBeNull();
  });
});
