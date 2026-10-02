/**
 * Which sub-box slot a crop card overlays and edits (domain-neutral audit
 * R1). A region lives on an item of a *different* class than the slot's
 * own bound class, so the card must pick the slot by the crop's region
 * evidence, not by `crop.class_name`. Before the fix, no caller passed a
 * slot, the class-name lookup never matched, the ring never drew, and the
 * ✎ button rendered on every card while its save silently no-oped. ✎ now
 * renders only when there is a sub-box slot to edit.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import type { Crop } from '$lib/types';
import type { SlotData, SlotSpec } from '$lib/annotations/types';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import { aircraftTailNumberSlot } from '$lib/test/fixtures/aircraftTailNumberSlot';
import { makeSlotBox } from '$lib/test/fixtures/slotBox';

const registry = vi.hoisted(() => ({ all: [] as SlotSpec[] }));

vi.mock('$lib/annotations/registeredSlots', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('$lib/annotations/registeredSlots')>();
  return {
    ...actual,
    get slotRegistry() {
      return { ...actual.slotRegistry, all: registry.all };
    },
  };
});

import CropCard from './CropCard.svelte';

function tagData(): SlotData {
  return {
    key: widgetTagSlot.key,
    subBoxes: [
      makeSlotBox({
        boxId: 'b1',
        state: 'accepted',
        rawXyxy: [0.4, 0.45, 0.6, 0.55],
        parent: { cx: 0.5, cy: 0.5, w: 0.2, h: 0.1 },
      }),
      makeSlotBox({
        boxId: 'b2',
        state: 'rejected',
        rawXyxy: [0.1, 0.1, 0.2, 0.2],
        parent: { cx: 0.15, cy: 0.15, w: 0.1, h: 0.1 },
      }),
    ],
    lifecycle: {
      status: 'detected',
      state: null,
      verified: true,
      validated: true,
      autoConfirmed: null,
      rejectionReason: null,
    },
  };
}

function widget(slots: Crop['slots'] = {}): Crop {
  return {
    id: 'w1',
    source_image_path: '/img.jpg',
    bbox_norm: { cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 },
    class_id: 7,
    class_name: 'widget_a',
    class_source: null,
    label_source: 'human',
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
    slots,
  } as Crop;
}

let target: HTMLDivElement;
let instance: unknown;

function renderCard(crop: Crop) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(CropCard, { target, props: { crop } } as never);
  flushSync();
  return target;
}

const editButton = (el: HTMLElement) =>
  [...el.querySelectorAll('button')].find((b) => b.textContent?.trim() === '✎');

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
  registry.all = [];
});

describe('CropCard — sub-box slot chosen by region evidence', () => {
  it("draws the region ring on an item whose own class isn't the slot's bound class", () => {
    registry.all = [widgetTagSlot];
    const el = renderCard(widget({ [widgetTagSlot.key]: tagData() }));

    // One ring per served box (two here), not just the first.
    expect(el.querySelectorAll('div.pointer-events-none.absolute')).toHaveLength(2);
    expect(editButton(el)?.getAttribute('aria-label')).toBe('Edit tag');
  });

  it('offers ✎ on an item with no region yet when exactly one sub-box slot is registered', () => {
    registry.all = [widgetTagSlot];
    const el = renderCard(widget());

    expect(el.querySelector('div.pointer-events-none.absolute')).toBeNull();
    expect(editButton(el)).toBeDefined();
  });

  it('renders no ✎ and no ring when no sub-box slot is registered', () => {
    registry.all = [];
    const el = renderCard(widget({ [widgetTagSlot.key]: tagData() }));

    expect(editButton(el)).toBeUndefined();
    expect(el.querySelector('div.pointer-events-none.absolute')).toBeNull();
  });

  it('offers no ✎ for a read-only scalar-box slot (the backend has no write route for it)', () => {
    registry.all = [aircraftTailNumberSlot];
    const el = renderCard(
      widget({
        [aircraftTailNumberSlot.key]: {
          key: aircraftTailNumberSlot.key,
          subBox: {
            rawXyxy: [0.4, 0.4, 0.6, 0.6],
            frame: 'parent',
            parent: { cx: 0.5, cy: 0.5, w: 0.2, h: 0.2 },
            score: 0.8,
            visible: true,
          },
        },
      }),
    );
    // Its box still draws.
    expect(el.querySelectorAll('div.pointer-events-none.absolute')).toHaveLength(1);
    expect(editButton(el)).toBeUndefined();
  });
});
