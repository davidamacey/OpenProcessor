/**
 * SlotBboxEditor lock glyph (W10): the editor's canvas marks a box whose
 * served `locked` is true (matched by box id off the crop's slot data) and
 * no other. Mounts the real editor.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import SlotBboxEditor from './SlotBboxEditor.svelte';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import { makeSlotBox } from '$lib/test/fixtures/slotBox';
import type { Crop } from '$lib/types';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function crop(): Crop {
  return {
    id: 'c1',
    source_image_path: '/img.jpg',
    bbox_norm: { cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 },
    class_id: 3,
    class_name: 'widget_tag',
    slots: {
      [widgetTagSlot.key]: {
        key: widgetTagSlot.key,
        subBoxes: [
          makeSlotBox({ boxId: 'b1', locked: true, rawXyxy: [0.1, 0.1, 0.3, 0.3] }),
          makeSlotBox({ boxId: 'b2', locked: false, rawXyxy: [0.5, 0.5, 0.7, 0.7] }),
          makeSlotBox({ boxId: 'b3', locked: null, rawXyxy: [0.2, 0.6, 0.4, 0.8] }),
        ],
        boxSet: {
          count: 3,
          rejectedCount: 0,
          maxScore: 0.9,
          setComplete: true,
          revision: 1,
        },
      },
    },
  } as unknown as Crop;
}

describe('SlotBboxEditor lock glyph', () => {
  it('marks exactly the locked boxes on the canvas', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(SlotBboxEditor, {
      target,
      props: { crop: crop(), slot: widgetTagSlot, onclose: () => {} },
    });
    flushSync();
    const rows = [...target.querySelectorAll('[data-box-index]')];
    expect(rows).toHaveLength(3);
    const locked = rows.map(
      (r) => r.querySelector('[data-testid="box-locked"]') !== null,
    );
    expect(locked).toEqual([true, false, false]);
  });
});
