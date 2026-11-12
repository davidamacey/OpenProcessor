import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { ImageBox } from '$lib/types_images';
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
    locked: false,
    lock_label: null,
    geometry_revision: null,
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
    expect(cls('a')).toContain('border-success');
    expect(cls('b')).toContain('border-warn');
    expect(cls('c')).toContain('border-danger');
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

  it('region boxes only for the selected item by default', () => {
    const boxes = [
      box({ crop_id: 'a' }),
      box({ crop_id: 'b' }),
      box({ crop_id: 'a', kind: 'region', box_id: 'r1' }),
      box({ crop_id: 'b', kind: 'region', box_id: 'r2' }),
    ];
    const el = render({ boxes, selectedCropId: 'a' });
    const regions = el.querySelectorAll('[data-kind="region"]');
    expect(regions).toHaveLength(1);
    expect(regions[0]!.getAttribute('data-crop-id')).toBe('a');
    expect(el.querySelectorAll('[data-kind="item"]')).toHaveLength(2);
  });

  it('draws every region box when allRegions is on', () => {
    const boxes = [
      box({ crop_id: 'a', kind: 'region', box_id: 'r1' }),
      box({ crop_id: 'b', kind: 'region', box_id: 'r2' }),
    ];
    const el = render({ boxes, selectedCropId: 'a', allRegions: true });
    expect(el.querySelectorAll('[data-kind="region"]')).toHaveLength(2);
  });

  it('labels every item box, not only the selected one', () => {
    const el = render({
      boxes: [box({ crop_id: 'a', label: 'cat' }), box({ crop_id: 'b', label: 'dog' })],
      selectedCropId: 'a',
    });
    expect(el.textContent).toContain('cat');
    expect(el.textContent).toContain('dog');
  });

  describe('spotlight', () => {
    const boxes = [
      box({ crop_id: 'sel', label: 'cat' }),
      box({ crop_id: 'other', label: 'dog', tone: 'machine' }),
      box({ crop_id: 'prop', label: 'bus', label_kind: 'detector', tone: 'unlabeled' }),
    ];
    const at = (el: HTMLElement, id: string) =>
      el.querySelector<HTMLElement>(`[data-crop-id="${id}"]`)!;

    it('draws the selected box 3 px in the highlight with the outside dimmed', () => {
      const el = render({ boxes, selectedCropId: 'sel', spotlight: true });
      const sel = at(el, 'sel');
      expect(sel.className).toContain('border-hl');
      expect(sel.className).toContain('border-3');
      expect(sel.className).toContain('dim-outside');
    });

    it('draws the others 1 px at 40% without labels, a detector proposal in the detector tone', () => {
      const el = render({ boxes, selectedCropId: 'sel', spotlight: true });
      const other = at(el, 'other');
      expect(other.className).toContain('border ');
      expect(other.className).not.toContain('border-2');
      expect(other.className).toContain('opacity-40');
      expect(at(el, 'prop').className).toContain('border-tone-detector');
      expect(el.textContent).not.toContain('dog');
      expect(el.textContent).not.toContain('bus');
    });

    it('does not dim or restyle without the flag', () => {
      const el = render({ boxes, selectedCropId: 'sel' });
      expect(at(el, 'sel').className).not.toContain('dim-outside');
      expect(at(el, 'other').className).toContain('border-2');
      expect(el.textContent).toContain('dog');
    });
  });
});
