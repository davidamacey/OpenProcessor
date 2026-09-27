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
    setSlotBox: vi.fn(async () => ({ id: 'c1' })),
  };
});

import ShortcutOverlay from './ShortcutOverlay.svelte';
import SlotBboxEditor from './SlotBboxEditor.svelte';
import { setSlotBox } from '$lib/api';
import { keyboardStore } from '$stores/keyboard.svelte';
import { keymapStore } from '$stores/keymap.svelte';
import { FALLBACK_KEYMAP, type KeymapDocument } from '$lib/keymapFallback';
import {
  installDeploymentSlots,
  resetDeploymentSlots,
} from '$lib/annotations/registeredSlots';
import { mapCropSlots } from '$lib/annotations/cropSlots';
import { aircraftTailNumberSlot } from '$lib/test/fixtures/aircraftTailNumberSlot';
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
    // aircraftTailNumberSlot is a genuine single-box (bboxField) slot —
    // the served region slot moved to the W8 multi-box list only (no
    // backward compatibility), so SlotBboxEditor (the legacy single-box
    // modal) is tested against this fixture instead.
    installDeploymentSlots([aircraftTailNumberSlot]);
    vi.mocked(setSlotBox).mockClear();
  });
  afterEach(() => resetDeploymentSlots());

  function crop(): Crop {
    return {
      id: 'c1',
      source_image_path: '/img.jpg',
      bbox_norm: { cx: 0.5, cy: 0.5, w: 1, h: 1 },
      class_id: 3,
      class_name: 'widget_a',
      label_validated: false,
      slots: mapCropSlots({ tail_bbox_norm: [0.4, 0.4, 0.6, 0.6] }, [0, 0, 1, 1]),
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
    const el = render(SlotBboxEditor, {
      crop: crop(),
      slot: aircraftTailNumberSlot,
      onclose: vi.fn(),
    });
    expect(footerKeys(el)).toEqual(['[', ']', '←↑↓→', '⌫', '↵', 'Esc']);
  });

  it('a rebound clear key renders and clears the box; the old key no longer does', async () => {
    keymapStore.setDocument(withKeys({ 'box_edit.delete_box': ['q'] }), 'served');
    const el = render(SlotBboxEditor, {
      crop: crop(),
      slot: aircraftTailNumberSlot,
      onclose: vi.fn(),
    });
    expect(footerKeys(el)).toEqual(['[', ']', '←↑↓→', 'Q', '↵', 'Esc']);

    // Old key: the box survives, so Enter saves a box.
    press({ key: 'Backspace' });
    press({ key: 'Enter' });
    await settle();
    expect(setSlotBox).toHaveBeenCalledTimes(1);
    expect(vi.mocked(setSlotBox).mock.calls[0][2]).not.toBeNull();

    // Rebound key: the box is cleared, so Enter saves "no box".
    press({ key: 'q' });
    press({ key: 'Enter' });
    await settle();
    expect(setSlotBox).toHaveBeenCalledTimes(2);
    expect(vi.mocked(setSlotBox).mock.calls[1][2]).toBeNull();
  });
});
