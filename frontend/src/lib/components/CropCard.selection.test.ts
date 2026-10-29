/**
 * The card is a container with exactly one select button; its other actions
 * are sibling buttons, never nested inside another interactive element
 * (axe `nested-interactive`).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import CropCard from './CropCard.svelte';
import type { Crop } from '$lib/types';

const crop = {
  id: 'c1',
  source_image_path: '/img.jpg',
  bbox_norm: { cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 },
  class_id: 3,
  class_name: 'widget_a',
  label_source: 'model_x',
  label_validated: false,
  class_validated: false,
  updated_at: '',
} as unknown as Crop;

let target: HTMLDivElement;
let instance: unknown;

function render(props: Record<string, unknown>): HTMLElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(CropCard, { target, props: { crop, ...props } } as never);
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) unmount(instance as never);
  instance = undefined;
  target?.remove();
});

describe('CropCard selection control', () => {
  it('has no role=button container and one select button named for the crop', () => {
    const el = render({ ondetail: () => {} });
    expect(el.querySelector('[role="button"]')).toBeNull();
    const selects = el.querySelectorAll('button[aria-pressed]');
    expect(selects).toHaveLength(1);
    expect(selects[0].getAttribute('aria-pressed')).toBe('false');
    expect(selects[0].getAttribute('aria-label')).toBe('Select crop c1');
  });

  it('nests no interactive element inside another', () => {
    const el = render({ ondetail: () => {} });
    const interactive = 'button, a[href], input, select, textarea, [role="button"]';
    for (const node of el.querySelectorAll(interactive)) {
      expect(node.parentElement?.closest(interactive)).toBeNull();
    }
  });

  it('reflects the selected state on the select button', () => {
    const el = render({ selected: true });
    expect(el.querySelector('button[aria-pressed]')?.getAttribute('aria-pressed')).toBe(
      'true',
    );
  });

  it('clicking the select button selects with the event', () => {
    const onclick = vi.fn();
    const el = render({ onclick });
    (el.querySelector('button[aria-pressed]') as HTMLButtonElement).click();
    expect(onclick).toHaveBeenCalledOnce();
    expect(onclick.mock.calls[0][0]).toBe(crop);
  });

  it('an inner action does not select the card', () => {
    const onclick = vi.fn();
    const ondetail = vi.fn();
    const el = render({ onclick, ondetail });
    (
      el.querySelector('button[aria-label="Show crop details"]') as HTMLButtonElement
    ).click();
    expect(ondetail).toHaveBeenCalledOnce();
    expect(onclick).not.toHaveBeenCalled();
  });
});
