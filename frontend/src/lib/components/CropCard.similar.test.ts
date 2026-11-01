/** The "More like this" action on a crop card. */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import CropCard from './CropCard.svelte';
import type { Crop } from '$lib/types';

function crop(overrides: Partial<Crop> = {}): Crop {
  return {
    id: 'c1',
    source_image_path: '/img.jpg',
    bbox_norm: { cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 },
    class_id: 3,
    class_name: 'widget_a',
    class_source: null,
    label_source: 'model_x',
    label_validated: false,
    class_validated: false,
    label_confidence: null,
    cluster_id: null,
    similarity_to_centroid: null,
    cluster_subid: null,
    proposed_class_id: null,
    proposed_class_name: null,
    test_holdout: false,
    updated_at: '',
    ...overrides,
  } as Crop;
}

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

function render(props: Record<string, unknown>): HTMLElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(CropCard, { target, props } as never);
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

const button = (el: HTMLElement) =>
  el.querySelector<HTMLButtonElement>('[data-testid="crop-similar"]');

describe('CropCard — More like this', () => {
  it('renders no action without onsimilar', () => {
    expect(button(render({ crop: crop() }))).toBeNull();
  });

  it('calls onsimilar once with the crop and does not fire the card onclick', () => {
    const onsimilar = vi.fn();
    const onclick = vi.fn();
    const c = crop();
    const el = render({ crop: c, onsimilar, onclick });
    const b = button(el);
    expect(b?.getAttribute('aria-label')).toBe('More like this');
    b?.click();
    expect(onsimilar).toHaveBeenCalledTimes(1);
    expect(onsimilar).toHaveBeenCalledWith(c);
    expect(onclick).not.toHaveBeenCalled();
  });

  it('is disabled, with the reason, for a crop the server says has no vector', () => {
    const onsimilar = vi.fn();
    const el = render({ crop: crop({ embedding_state: 'deferred' }), onsimilar });
    const b = button(el);
    expect(b?.disabled).toBe(true);
    expect(b?.getAttribute('title')).toMatch(/vector/i);
    b?.click();
    expect(onsimilar).not.toHaveBeenCalled();
  });
});
