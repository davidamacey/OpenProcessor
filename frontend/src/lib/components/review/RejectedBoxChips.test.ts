import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { createMultiBoxRegionController } from '$lib/review/multiBoxRegionController.svelte';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import RejectedBoxChips from './RejectedBoxChips.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function render(boxes: ReturnType<typeof createMultiBoxRegionController>['boxes']) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(RejectedBoxChips, { target, props: { boxes, subBoxes: [] } });
  flushSync();
}

describe('RejectedBoxChips', () => {
  it('shows one chip per rejected box when two unsaved boxes are rejected with r', async () => {
    const c = createMultiBoxRegionController(() => widgetTagSlot);
    c.seedFrom(null);
    c.addBox({ cx: 0.2, cy: 0.2, w: 0.1, h: 0.1 });
    await c.rejectSelected('crop-1');
    c.addBox({ cx: 0.6, cy: 0.6, w: 0.1, h: 0.1 });
    await c.rejectSelected('crop-1');
    expect(c.boxes.map((b) => [b.boxId, b.state])).toEqual([
      [null, 'rejected'],
      [null, 'rejected'],
    ]);
    render(c.boxes);
    expect(target.querySelectorAll('span').length).toBe(2);
  });

  it('control: a single rejected box and an accepted one give one chip', () => {
    render([
      { boxId: 'b1', state: 'rejected' },
      { boxId: 'b2', state: 'accepted' },
    ] as never);
    expect(target.querySelectorAll('span').length).toBe(1);
  });
});
