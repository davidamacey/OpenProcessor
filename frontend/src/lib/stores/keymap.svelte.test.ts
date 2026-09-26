/**
 * keymapStore (configurable-keyboard-shortcuts plan §5.1, step K1): the
 * fallback document reproduces today's keys, ids resolve to combos and
 * labels, contexts resolve to their registrations, and the locked-key
 * invariant holds for any document handed to `setDocument`.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { keymapStore } from './keymap.svelte';
import { FALLBACK_KEYMAP, type KeymapDocument } from '$lib/keymapFallback';
import { classesStore } from './classes.svelte';
import { reservedHotkeyLetters } from '$lib/classHotkey';

/** The fallback document with some actions' keys replaced. */
function withKeys(overrides: Record<string, string[]>): KeymapDocument {
  return {
    ...FALLBACK_KEYMAP,
    actions: FALLBACK_KEYMAP.actions.map((a) =>
      a.id in overrides ? { ...a, keys: overrides[a.id] } : a,
    ),
  };
}

afterEach(() => {
  keymapStore.resetToFallback();
  vi.restoreAllMocks();
});

describe('FALLBACK_KEYMAP', () => {
  it('declares the 46 action ids of plan §1.1, each once', () => {
    const ids = FALLBACK_KEYMAP.actions.map((a) => a.id);
    expect(ids).toHaveLength(46);
    expect(new Set(ids).size).toBe(46);
  });

  it("reproduces today's default keys", () => {
    const expected: Record<string, string[]> = {
      'global.shortcuts_overlay': ['~', '`', 'shift+~'],
      'global.close_overlay': ['escape'],
      'review.skip': ['n'],
      'review.undo': ['z'],
      'review.queue.confirm': ['enter'],
      'review.queue.discard': ['d'],
      'review.queue.class_picker': ['/'],
      'review.queue.prev': ['arrowleft'],
      'review.queue.next': ['arrowright'],
      'review.region.confirm': ['enter'],
      'review.region.reject': ['d'],
      'review.region.false_positive': ['f'],
      'review.region.edit_box': ['e'],
      'review.region.back': ['arrowleft', 'b'],
      'review.region.next': ['arrowright'],
      'box_edit.save': ['enter'],
      'box_edit.cancel': ['escape'],
      'box_edit.nudge_up': ['arrowup'],
      'box_edit.nudge_down': ['arrowdown'],
      'box_edit.nudge_left': ['arrowleft'],
      'box_edit.nudge_right': ['arrowright'],
      'box_edit.shrink_right': ['['],
      'box_edit.grow_right': [']'],
      'box_edit.delete_box': ['backspace'],
      'cluster.confirm': ['enter'],
      'cluster.accept_all_vlm': ['shift+enter'],
      'cluster.accept_vlm': ['g'],
      'cluster.skip': ['n'],
      'cluster.flag_new_class': ['shift+n'],
      'cluster.discard': ['d'],
      'cluster.undo': ['z'],
      'cluster.ignore': ['x'],
      'cluster.unignore': ['u'],
      'cluster.select_all': ['a'],
      'cluster.prev': ['arrowleft'],
      'cluster.next': ['arrowright'],
      'cluster.move': ['m'],
      'cluster.cancel': ['escape'],
      'clusters_search.select_all': ['a'],
      'clusters_search.cancel': ['escape'],
      'clusters_search.ignore': ['x'],
      'clusters_search.undo': ['z'],
      'region_gallery.undo': ['z'],
    };
    for (const [id, keys] of Object.entries(expected)) {
      expect(keymapStore.keysFor(id), id).toEqual(keys);
    }
  });

  it('has no unavailable actions — the W8 per-box actions are enabled on this branch', () => {
    const unavailable = FALLBACK_KEYMAP.actions
      .filter((a) => !a.available)
      .map((a) => a.id);
    expect(unavailable).toEqual([]);
  });
});

describe('resolving an action id', () => {
  it('formats the first combo as the glyph, and every combo as glyphs', () => {
    expect(keymapStore.glyph('review.region.back')).toBe('←');
    expect(keymapStore.glyphs('review.region.back')).toEqual(['←', 'B']);
    expect(keymapStore.glyph('cluster.flag_new_class')).toBe('Shift+N');
    expect(keymapStore.glyph('global.shortcuts_overlay')).toBe('~');
  });

  it('has a compact glyph for tight spaces', () => {
    expect(keymapStore.compactGlyph('cluster.flag_new_class')).toBe('⇧N');
    expect(keymapStore.compactGlyph('cluster.accept_all_vlm')).toBe('⇧↵');
    expect(keymapStore.compactGlyph('box_edit.delete_box')).toBe('⌫');
    expect(keymapStore.compactGlyph('box_edit.cancel')).toBe('Esc');
  });

  it('fills the region noun into a label, defaulting to "region"', () => {
    expect(keymapStore.label('review.region.reject', { region: 'widget tag' })).toBe(
      'Reject (no widget tag visible)',
    );
    expect(keymapStore.label('review.region.reject')).toBe('Reject (no region visible)');
  });

  it('returns no keys and an empty glyph for an unknown id', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(keymapStore.keysFor('review.nope')).toEqual([]);
    expect(keymapStore.glyph('review.nope')).toBe('');
  });

  it('follows a new document with no re-registration', () => {
    keymapStore.setDocument(withKeys({ 'review.queue.discard': ['x'] }), 'served');
    expect(keymapStore.keysFor('review.queue.discard')).toEqual(['x']);
    expect(keymapStore.glyph('review.queue.discard')).toBe('X');
  });
});

describe('resolving a context', () => {
  it('lists only the available actions declared in that context', () => {
    const ids = keymapStore.actionsForContext('review.region').map((a) => a.id);
    expect(ids).toEqual([
      'review.region.confirm',
      'review.region.reject',
      'review.region.false_positive',
      'review.region.edit_box',
      'review.region.back',
      'review.region.next',
      'review.region.accept_box',
      'review.region.reject_box',
    ]);
  });

  it('walks includes transitively: review.region sees review and global keys', () => {
    expect(keymapStore.activeSet('review.region')).toEqual([
      'review.region',
      'review',
      'global',
    ]);
    expect(keymapStore.actionFor('review.region', 'z')).toBe('review.undo');
    expect(keymapStore.actionFor('review.region', '`')).toBe('global.shortcuts_overlay');
  });

  it("prefers the context's own action over an inherited one", () => {
    expect(keymapStore.actionFor('box_edit', 'escape')).toBe('box_edit.cancel');
    expect(keymapStore.actionFor('box_edit', 'enter')).toBe('box_edit.save');
  });

  it('resolves the W8 per-box actions now that this branch enables them', () => {
    expect(keymapStore.actionFor('box_edit', 'tab')).toBe('box_edit.next_box');
    expect(keymapStore.actionFor('review.region', 'y')).toBe('review.region.accept_box');
    expect(keymapStore.actionFor('review.region', 'r')).toBe('review.region.reject_box');
  });

  it('does not see a sibling context: box_edit excludes review', () => {
    expect(keymapStore.actionFor('box_edit', 'n')).toBeNull();
  });
});

describe('locked-key invariant', () => {
  it('a locked action keeps its locked key when a document drops it', () => {
    keymapStore.setDocument(withKeys({ 'review.queue.confirm': ['c'] }), 'served');
    expect(keymapStore.keysFor('review.queue.confirm')).toEqual(['enter', 'c']);
    expect(keymapStore.actionFor('review.queue', 'enter')).toBe('review.queue.confirm');
  });

  it('keeps every locked key of a multi-key action', () => {
    keymapStore.setDocument(withKeys({ 'review.region.back': ['k'] }), 'served');
    expect(keymapStore.keysFor('review.region.back')).toEqual(['arrowleft', 'k']);
  });

  it('no other action may take a locked key', () => {
    keymapStore.setDocument(
      withKeys({ 'review.queue.discard': ['escape', 'x'], 'cluster.move': ['enter'] }),
      'served',
    );
    expect(keymapStore.keysFor('review.queue.discard')).toEqual(['x']);
    expect(keymapStore.keysFor('cluster.move')).toEqual([]);
  });

  it('an unmodifiable action ignores the document and keeps its default', () => {
    keymapStore.setDocument(withKeys({ 'global.close_overlay': ['q'] }), 'served');
    expect(keymapStore.keysFor('global.close_overlay')).toEqual(['escape']);
  });

  it('caps an action at max_combos_per_action without dropping a locked key', () => {
    keymapStore.setDocument(
      withKeys({ 'review.region.back': ['h', 'j', 'k', 'l'] }),
      'served',
    );
    expect(keymapStore.keysFor('review.region.back')).toEqual(['arrowleft', 'h', 'j']);
  });
});

describe('document handling', () => {
  it('ignores a served id this build does not know', () => {
    const doc = withKeys({});
    doc.actions = [
      ...doc.actions,
      { ...doc.actions[0], id: 'future.thing', keys: ['q'] },
    ];
    keymapStore.setDocument(doc, 'served');
    expect(keymapStore.action('future.thing')).toBeUndefined();
  });

  it('falls back to the default for an id the served document omits, with a warning', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    keymapStore.setDocument(
      {
        ...FALLBACK_KEYMAP,
        actions: FALLBACK_KEYMAP.actions.filter((a) => a.id !== 'cluster.move'),
      },
      'served',
    );
    expect(keymapStore.keysFor('cluster.move')).toEqual(['m']);
    expect(warn).toHaveBeenCalled();
  });
});

describe('reserved hotkeys', () => {
  it('is null on the fallback, so the served classes reserved_hotkeys stays authoritative', () => {
    expect(keymapStore.reserved).toBeNull();
    const prev = classesStore.reservedHotkeys;
    classesStore.reservedHotkeys = ['q'];
    try {
      expect(reservedHotkeyLetters().has('q')).toBe(true);
    } finally {
      classesStore.reservedHotkeys = prev;
    }
  });

  it("uses a served keymap's reserved set when there is one", () => {
    keymapStore.setDocument({ ...FALLBACK_KEYMAP, reserved_hotkeys: ['k'] }, 'served');
    expect(keymapStore.reserved).toEqual(['k']);
    const prev = classesStore.reservedHotkeys;
    classesStore.reservedHotkeys = ['q'];
    try {
      const reserved = reservedHotkeyLetters();
      expect(reserved.has('k')).toBe(true);
      expect(reserved.has('q')).toBe(false);
    } finally {
      classesStore.reservedHotkeys = prev;
    }
  });
});
