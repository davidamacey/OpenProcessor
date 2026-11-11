import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import HiddenItemsNote from './HiddenItemsNote.svelte';

let instance: ReturnType<typeof mount> | null = null;
let target: HTMLDivElement;

function render(hidden: { test_holdout: number; excluded: number } | null) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(HiddenItemsNote, { target, props: { hidden } });
  flushSync();
  return target.querySelector('[data-testid="hidden-items-note"]');
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
});

describe('HiddenItemsNote', () => {
  it('says what is left out, from the served counts', () => {
    expect(render({ test_holdout: 25, excluded: 4 })?.textContent).toContain(
      '25 test holdout and 4 ignored not listed',
    );
  });

  it('names only the kind that is hidden', () => {
    expect(render({ test_holdout: 25, excluded: 0 })?.textContent).toContain(
      '25 test holdout not listed',
    );
  });

  it('renders nothing when nothing is hidden or the page did not say', () => {
    expect(render({ test_holdout: 0, excluded: 0 })).toBeNull();
    unmount(instance!);
    target.remove();
    expect(render(null)).toBeNull();
  });
});
