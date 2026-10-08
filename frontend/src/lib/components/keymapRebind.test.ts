/**
 * End-to-end proof that a rebound keymap changes real UI and real key
 * handling (configurable-keyboard-shortcuts plan §5.6, step K1): swap the
 * keymap document, then mount the real components and check both the
 * printed key and which physical key actually triggers the action.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';

vi.mock('$lib/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api')>();
  return {
    ...actual,
    getThumbUrl: () => '',
    putRegionBoxes: vi.fn(async () => ({ id: 'c1' })),
  };
});

import ShortcutOverlay from './ShortcutOverlay.svelte';
import SlotBboxEditor from './SlotBboxEditor.svelte';
import { putRegionBoxes } from '$lib/api';
import { keyboardStore } from '$stores/keyboard.svelte';
import { keymapStore } from '$stores/keymap.svelte';
import { FALLBACK_KEYMAP, type KeymapDocument } from '$lib/keymapFallback';
import {
  installServedRegionProfile,
  resetDeploymentSlots,
  slotForClassName,
} from '$lib/annotations/registeredSlots';
import { mapCropSlots } from '$lib/annotations/cropSlots';
import { WIDGET_TAG_CLASS, WIDGET_TAG_PROFILE } from '$lib/test/fixtures/regionSlot';
import type { Crop } from '$lib/types';

function withKeys(overrides: Record<string, string[]>): KeymapDocument {
  return {
    ...FALLBACK_KEYMAP,
    actions: FALLBACK_KEYMAP.actions.map((a) =>
      a.id in overrides ? { ...a, keys: overrides[a.id] } : a,
    ),
  };
}

function press(init: KeyboardEventInit & { key: string }): void {
  window.dispatchEvent(
    new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init }),
  );
}

let target: HTMLDivElement;
let instance: unknown;
const cleanups: Array<() => void> = [];

function render(component: unknown, props: Record<string, unknown>): HTMLDivElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(component as never, { target, props } as never);
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) unmount(instance as never);
  instance = undefined;
  target?.remove();
  while (cleanups.length) cleanups.pop()!();
  keymapStore.resetToFallback();
  keyboardStore.closeOverlay();
  keyboardStore.setScope('global');
});

describe('ShortcutOverlay prints the keymap', () => {
  /** The overlay row for `label`: [label, printed key]. */
  function row(el: HTMLElement, label: string): string[] | undefined {
    for (const li of el.querySelectorAll('li')) {
      const spans = li.querySelector('span');
      if (spans?.textContent?.trim() === label) {
        return [label, li.querySelector('kbd')?.textContent?.trim() ?? ''];
      }
    }
    return undefined;
  }

  it('shows the default keys', () => {
    keyboardStore.setScope('review');
    cleanups.push(
      keyboardStore.registerAction('review.queue.discard', vi.fn(), 'review'),
    );
    keyboardStore.toggleOverlay();
    const el = render(ShortcutOverlay, {});
    expect(row(el, 'Discard')).toEqual(['Discard', 'D']);
    expect(row(el, 'Toggle this panel')).toEqual(['Toggle this panel', '~']);
  });

  it('shows a rebound key, and the rebound key fires while the old one does not', () => {
    keymapStore.setDocument(
      withKeys({
        'review.queue.discard': ['x'],
        'global.shortcuts_overlay': ['shift+/'],
      }),
      'served',
    );
    const discard = vi.fn();
    keyboardStore.setScope('review');
    cleanups.push(
      keyboardStore.registerAction('review.queue.discard', discard, 'review'),
    );
    keyboardStore.toggleOverlay();
    const el = render(ShortcutOverlay, {});

    expect(row(el, 'Discard')).toEqual(['Discard', 'X']);
    expect(row(el, 'Toggle this panel')).toEqual(['Toggle this panel', 'Shift+/']);

    keyboardStore.closeOverlay();
    press({ key: 'd' });
    expect(discard).not.toHaveBeenCalled();
    press({ key: 'x' });
    expect(discard).toHaveBeenCalledTimes(1);
  });
});

describe('SlotBboxEditor resolves its keys through the keymap', () => {
  beforeEach(() => {
    installServedRegionProfile(WIDGET_TAG_PROFILE);
    vi.mocked(putRegionBoxes).mockClear();
  });
  afterEach(() => resetDeploymentSlots());

  const slot = () => slotForClassName(WIDGET_TAG_CLASS)!;

  function crop(): Crop {
    return {
      id: 'c1',
      source_image_path: '/img.jpg',
      bbox_norm: { cx: 0.5, cy: 0.5, w: 1, h: 1 },
      class_id: 3,
      class_name: 'widget_a',
      label_validated: false,
      slots: mapCropSlots(
        {
          region_boxes: [
            {
              box_id: 'b1',
              state: 'accepted',
              bbox_norm: [0.4, 0.4, 0.6, 0.6],
              bbox_in_parent: [0.4, 0.4, 0.6, 0.6],
            },
          ],
          region_revision: 2,
        },
        [0, 0, 1, 1],
      ),
    } as unknown as Crop;
  }

  function footerKeys(el: HTMLElement): string[] {
    return [...el.querySelectorAll('footer kbd')].map((k) => k.textContent?.trim() ?? '');
  }

  async function settle(): Promise<void> {
    await Promise.resolve();
    await Promise.resolve();
    flushSync();
  }

  it('prints the default box-edit keys', () => {
    const el = render(SlotBboxEditor, { crop: crop(), slot: slot(), onclose: vi.fn() });
    expect(footerKeys(el)).toEqual(['Tab', '⌫', '←↑↓→', '↵', 'Esc']);
  });

  it('a rebound delete key renders and removes the selected box; the old key no longer does', async () => {
    keymapStore.setDocument(withKeys({ 'box_edit.delete_box': ['q'] }), 'served');
    const el = render(SlotBboxEditor, { crop: crop(), slot: slot(), onclose: vi.fn() });
    expect(footerKeys(el)).toEqual(['Tab', 'Q', '←↑↓→', '↵', 'Esc']);

    // Old key: the box survives, so Enter saves the box list unchanged.
    press({ key: 'Backspace' });
    press({ key: 'Enter' });
    await settle();
    expect(putRegionBoxes).toHaveBeenCalledTimes(1);
    expect(vi.mocked(putRegionBoxes).mock.calls[0][1]).toEqual([{ box_id: 'b1' }]);
  });

  it('the rebound key deletes the box, so Save sends an empty list', async () => {
    keymapStore.setDocument(withKeys({ 'box_edit.delete_box': ['q'] }), 'served');
    render(SlotBboxEditor, { crop: crop(), slot: slot(), onclose: vi.fn() });
    press({ key: 'q' });
    press({ key: 'Enter' });
    await settle();
    expect(putRegionBoxes).toHaveBeenCalledTimes(1);
    expect(vi.mocked(putRegionBoxes).mock.calls[0][1]).toEqual([]);
    // The served revision rides along as the optimistic-concurrency guard.
    expect(vi.mocked(putRegionBoxes).mock.calls[0][2]).toMatchObject({
      expectedRegionRevision: 2,
    });
  });
});
