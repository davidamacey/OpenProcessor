import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { ImageBox } from '$lib/types';
import BoxLayer from './BoxLayer.svelte';

function box(over: Partial<ImageBox> = {}): ImageBox {
  return {
    crop_id: 'c1',
    box_id: null,
    kind: 'item',
    bbox_norm: [0.1, 0.2, 0.5, 0.6],
    label: 'widget',
    label_kind: 'class',
    class_name: 'widget',
    tone: 'validated',
    state: null,
    ...over,
  };
}

let instance: Record<string, unknown> | undefined;
let target: HTMLDivElement;

function render(props: Record<string, unknown>): HTMLDivElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(BoxLayer, { target, props: props as never });
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

describe('BoxLayer', () => {
  it('draws one rect per served box at served coordinates', () => {
    const el = render({
      boxes: [box(), box({ crop_id: 'c2', bbox_norm: [0.6, 0.6, 0.9, 0.8] })],
      selectedCropId: null,
    });
    const rects = el.querySelectorAll<HTMLElement>('[data-testid="overlay-box"]');
    expect(rects).toHaveLength(2);
    expect(rects[0]!.style.left).toBe('10%');
    expect(rects[0]!.style.top).toBe('20%');
    expect(rects[0]!.style.width).toBe('40%');
    expect(rects[1]!.style.left).toBe('60%');
  });

  it('shows the served label verbatim', () => {
    const el = render({
      boxes: [box({ label: 'person', label_kind: 'detector' })],
      selectedCropId: 'c1',
    });
    expect(el.querySelector('[data-testid="overlay-box"]')!.getAttribute('title')).toBe(
      'person',
    );
    expect(el.textContent).toContain('person');
  });

  it('one class per tone', () => {
    const el = render({
      boxes: [
        box({ crop_id: 'a', tone: 'validated' }),
        box({ crop_id: 'b', tone: 'machine' }),
        box({ crop_id: 'c', tone: 'rejected' }),
      ],
      selectedCropId: null,
    });
    const cls = (id: string) => el.querySelector(`[data-crop-id="${id}"]`)!.className;
    expect(cls('a')).toContain('border-emerald-400');
    expect(cls('b')).toContain('border-amber-400');
    expect(cls('c')).toContain('border-red-400');
  });

  it('rings the selected item and fires onselect only for the others', () => {
    const onselect = vi.fn();
    const el = render({
      boxes: [box({ crop_id: 'a' }), box({ crop_id: 'b' })],
      selectedCropId: 'a',
      onselect,
    });
    const a = el.querySelector<HTMLElement>('[data-crop-id="a"]')!;
    const b = el.querySelector<HTMLElement>('[data-crop-id="b"]')!;
    expect(a.className).toContain('ring-2');
    expect(a.tagName).toBe('DIV');
    expect(b.tagName).toBe('BUTTON');
    b.click();
    expect(onselect).toHaveBeenCalledWith('b');
  });
});
