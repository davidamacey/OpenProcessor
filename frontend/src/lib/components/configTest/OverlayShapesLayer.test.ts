import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { OverlayShape } from '$lib/configTest/overlayShapes';
import OverlayShapesLayer from './OverlayShapesLayer.svelte';

let instance: ReturnType<typeof mount> | undefined;
let target: HTMLElement | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  target?.remove();
  instance = undefined;
});

const shapes: OverlayShape[] = [
  { key: 'a', kind: 'box', box: [0.1, 0.2, 0.5, 0.6], dimmed: false, title: 'kept' },
  { key: 'b', kind: 'box', box: [0, 0, 1, 1], dimmed: true, title: 'dropped' },
  {
    key: 'c',
    kind: 'polygon',
    points: [
      [0, 0],
      [1, 0],
      [1, 1],
    ],
    dimmed: false,
    title: 'outline',
  },
] as OverlayShape[];

describe('OverlayShapesLayer', () => {
  it('draws boxes and outlines as served with the caller test ids, dropped ones faint', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(OverlayShapesLayer, {
      target,
      props: { shapes, boxTestid: 'x-box', polygonTestid: 'x-poly' },
    });
    flushSync();
    const boxes = [...target.querySelectorAll('[data-testid="x-box"]')];
    expect(boxes).toHaveLength(2);
    expect(boxes[0]!.getAttribute('width')).toBe('0.4');
    expect(boxes[0]!.getAttribute('data-dimmed')).toBe('false');
    expect(boxes[1]!.getAttribute('data-dimmed')).toBe('true');
    expect(boxes[1]!.getAttribute('opacity')).toBe('0.4');
    expect(target.querySelector('[data-testid="x-poly"]')!.getAttribute('points')).toBe(
      '0,0 1,0 1,1',
    );
  });
});
