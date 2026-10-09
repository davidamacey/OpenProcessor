/** box_edit.shrink_right / grow_right ([ and ]) move the selected box's right edge. */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import MultiBoxCanvas from './MultiBoxCanvas.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;
afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function render(onmove: (i: number, b: unknown) => void) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(MultiBoxCanvas, {
    target,
    props: {
      cropId: 'c_1',
      boxes: [
        { box: { cx: 0.5, cy: 0.5, w: 0.2, h: 0.2 }, state: 'accepted', label: 'x' },
      ],
      selectedIndex: 0,
      readonly: false,
      onmove,
    },
  });
  flushSync();
  return instance as unknown as { handleKey(e: KeyboardEvent): boolean };
}

describe('MultiBoxCanvas right-edge keys', () => {
  it('] grows the right edge and keeps the left edge', () => {
    const onmove = vi.fn();
    const c = render(onmove);
    expect(c.handleKey(new KeyboardEvent('keydown', { key: ']' }))).toBe(true);
    const [, b] = onmove.mock.calls[0]! as [number, { cx: number; w: number }];
    expect(b.w).toBeGreaterThan(0.2);
    expect(b.cx - b.w / 2).toBeCloseTo(0.4);
  });
  it('[ shrinks the right edge and keeps the left edge', () => {
    const onmove = vi.fn();
    const c = render(onmove);
    expect(c.handleKey(new KeyboardEvent('keydown', { key: '[' }))).toBe(true);
    const [, b] = onmove.mock.calls[0]! as [number, { cx: number; w: number }];
    expect(b.w).toBeLessThan(0.2);
    expect(b.cx - b.w / 2).toBeCloseTo(0.4);
  });
  it('control: an arrow still nudges (stays green before and after)', () => {
    const onmove = vi.fn();
    const c = render(onmove);
    expect(c.handleKey(new KeyboardEvent('keydown', { key: 'ArrowRight' }))).toBe(true);
    expect(onmove).toHaveBeenCalledTimes(1);
  });
});
