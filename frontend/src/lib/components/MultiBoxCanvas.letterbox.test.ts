/**
 * The thumbnail keeps its aspect ratio, so the box layer and the pointer
 * mapping must use the image's own rect, not the square container around it.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import MultiBoxCanvas from './MultiBoxCanvas.svelte';
import type { BBoxNormLike } from '$lib/annotations/types';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function render(onadd: (b: BBoxNormLike) => void) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(MultiBoxCanvas, {
    target,
    props: {
      cropId: 'c_1',
      boxes: [
        { box: { cx: 0.5, cy: 0.5, w: 0.2, h: 0.2 }, state: 'accepted', label: 'b' },
      ],
      selectedIndex: null,
      onadd,
    },
  });
  flushSync();
  const img = target.querySelector('img') as HTMLImageElement;
  Object.defineProperty(img, 'naturalWidth', { value: 200 });
  Object.defineProperty(img, 'naturalHeight', { value: 100 });
  img.dispatchEvent(new Event('load'));
  flushSync();
  return img;
}

describe('MultiBoxCanvas letterboxing', () => {
  it('sizes the image box to the image aspect ratio and fills it with the image', () => {
    const img = render(() => {});
    const frame = target.querySelector('[data-testid="multibox-canvas"]') as HTMLElement;
    expect(frame.style.aspectRatio.replace(/\s/g, '')).toBe('200/100');
    expect(img.className).toContain('object-fill');
    expect(frame.contains(target.querySelector('[data-box-index]'))).toBe(true);
  });

  it('maps pointer positions against the image frame, not the square', () => {
    const added: BBoxNormLike[] = [];
    render((b) => added.push(b));
    const frame = target.querySelector('[data-testid="multibox-canvas"]') as HTMLElement;
    // Square is 100x100; a 2:1 image fills it as 100x50 starting at y=25.
    frame.getBoundingClientRect = () =>
      ({ left: 0, top: 25, width: 100, height: 50, right: 100, bottom: 75 }) as DOMRect;
    frame.setPointerCapture = () => {};
    const ev = (type: string, x: number, y: number) =>
      Object.assign(new Event(type, { bubbles: true }), {
        clientX: x,
        clientY: y,
        pointerId: 1,
      });
    frame.dispatchEvent(ev('pointerdown', 0, 25));
    frame.dispatchEvent(ev('pointermove', 100, 75));
    frame.dispatchEvent(ev('pointerup', 100, 75));
    expect(added).toHaveLength(1);
    expect(added[0]).toMatchObject({ cx: 0.5, cy: 0.5, w: 1, h: 1 });
  });
});
