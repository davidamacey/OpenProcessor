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

describe('ReviewItemMeta Current label', () => {
  it('names an unclassed item by the served display name, not "no class yet"', () => {
    const el = render({
      class_id: null,
      class_name: '',
      display_name: 'stop sign',
      display_name_kind: 'detector',
    });
    const row = el.querySelector('[data-testid="current-label"]')!;
    expect(row.textContent).toContain('stop sign');
    expect(row.textContent).toContain("detector's label");
    expect(row.textContent).not.toContain('no class yet');
  });

  it('says "no class yet" only when the server names nothing', () => {
    const el = render({
      class_id: null,
      class_name: '',
      display_name: '',
      display_name_kind: 'none',
    });
    expect(el.querySelector('[data-testid="current-label"]')!.textContent).toContain(
      'no class yet',
    );
  });

  it('shows the class name of a classed item', () => {
    const el = render({ display_name: 'widget_a', display_name_kind: 'class' });
    expect(el.querySelector('[data-testid="current-label"]')!.textContent).toContain(
      'widget_a',
    );
  });
});
