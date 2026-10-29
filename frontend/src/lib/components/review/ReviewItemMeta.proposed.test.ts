/** The Proposed row exists only when the server served a proposal. */
import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ReviewItemMeta from './ReviewItemMeta.svelte';
import type { ReviewItem } from '$lib/types_review';

let target: HTMLDivElement;
let instance: unknown;

function render(overrides: Partial<ReviewItem>): HTMLElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  const current = {
    id: 'c1',
    class_id: 1,
    class_name: 'widget_a',
    label_source: 'model_x',
    reason: 'uncertainty',
    ...overrides,
  } as unknown as ReviewItem;
  instance = mount(ReviewItemMeta, {
    target,
    props: {
      current,
      activeSlot: null,
      isMultiBoxSlot: false,
      opinion: { kind: 'none', showAccept: false },
      searchScores: new Map(),
      acceptModelClass: async () => {},
    },
  } as never);
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) unmount(instance as never);
  instance = undefined;
  target?.remove();
});

const labels = (el: HTMLElement) =>
  [...el.querySelectorAll('dt')].map((d) => d.textContent?.trim());

describe('ReviewItemMeta Proposed row', () => {
  it('is absent when there is no proposal', () => {
    const el = render({ proposed_class_id: null, proposed_class_name: null });
    expect(labels(el)).not.toContain('Proposed');
  });

  it('shows the served proposal', () => {
    const el = render({ proposed_class_id: 2, proposed_class_name: 'widget_b' });
    expect(labels(el)).toContain('Proposed');
    expect(el.textContent).toContain('widget_b');
  });

  it('shows a name-only proposal as a hint', () => {
    const el = render({ proposed_class_id: null, proposed_class_name: 'odd_term' });
    expect(labels(el)).toContain('Proposed');
    expect(el.textContent).toContain('hint only');
  });
});
