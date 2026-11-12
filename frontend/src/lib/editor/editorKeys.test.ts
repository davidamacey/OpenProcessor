import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FALLBACK_KEYMAP, type KeymapDocument } from '$lib/keymapFallback';
import type { RegistryClass } from '$lib/types_classes';
import { dropOnClassStore } from '$stores/dropOnClass.svelte';
import { keyboardStore } from '$stores/keyboard.svelte';
import { keymapStore } from '$stores/keymap.svelte';
import type { EditorController } from './editorController.svelte';
import { EDITOR_SCOPE, registerEditorKeys } from './editorKeys.svelte';

function fakeController(over: Partial<Record<string, unknown>> = {}) {
  const fn = () => vi.fn();
  const c = {
    selected: { id: 'c1' },
    objects: [{ cropId: 'c1' }],
    setTool: fn(),
    zoomIn: fn(),
    zoomOut: fn(),
    zoomFit: fn(),
    zoomActual: fn(),
    deleteSelected: fn(),
    selectNext: fn(),
    selectPrev: fn(),
    editRegions: fn(),
    toggleLabels: fn(),
    undo: fn(),
    redo: fn(),
    openClassPicker: fn(),
    prevImage: fn(),
    nextImage: fn(),
    escape: fn(),
    nudge: fn(),
    nudgeFar: fn(),
    resizeCorner: fn(),
    setClass: fn(),
    ...over,
  };
  return c as unknown as EditorController & typeof c;
}

function press(init: KeyboardEventInit & { key: string }): KeyboardEvent {
  const e = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init });
  window.dispatchEvent(e);
  return e;
}

let dispose: (() => void) | null = null;

function withKeys(overrides: Record<string, string[]>): KeymapDocument {
  return {
    ...FALLBACK_KEYMAP,
    actions: FALLBACK_KEYMAP.actions.map((a) =>
      a.id in overrides ? { ...a, keys: overrides[a.id] } : a,
    ),
  };
}

beforeEach(() => keymapStore.resetToFallback());
afterEach(() => {
  dispose?.();
  dispose = null;
  keymapStore.resetToFallback();
  keyboardStore.setScope('global');
});

describe('editor keys', () => {
  it('binds every editor action the keymap serves', () => {
    dispose = registerEditorKeys(fakeController());
    const unbound = FALLBACK_KEYMAP.actions
      .filter((a) => a.context === 'editor')
      .filter((a) => !a.keys.some((k) => keyboardStore.hasActiveBinding(k)))
      .map((a) => a.id);
    expect(unbound).toEqual([]);
  });

  it('runs the tool, undo and navigation actions from their default keys', () => {
    const c = fakeController();
    dispose = registerEditorKeys(c);
    press({ key: 'v' });
    expect(c.setTool).toHaveBeenLastCalledWith('select');
    press({ key: 'b' });
    expect(c.setTool).toHaveBeenLastCalledWith('draw');
    press({ key: 'B', shiftKey: true });
    expect(c.setTool).toHaveBeenLastCalledWith('draw', true);
    press({ key: 'h' });
    expect(c.setTool).toHaveBeenLastCalledWith('pan');
    press({ key: 'z', ctrlKey: true });
    expect(c.undo).toHaveBeenCalledOnce();
    press({ key: 'Z', ctrlKey: true, shiftKey: true });
    expect(c.redo).toHaveBeenCalledOnce();
    press({ key: 'z', metaKey: true });
    expect(c.undo).toHaveBeenCalledTimes(2);
    press({ key: '.' });
    expect(c.nextImage).toHaveBeenCalledOnce();
    press({ key: ',' });
    expect(c.prevImage).toHaveBeenCalledOnce();
    press({ key: '/' });
    expect(c.openClassPicker).toHaveBeenCalledOnce();
    press({ key: 'Escape' });
    expect(c.escape).toHaveBeenCalledOnce();
    press({ key: 'L', shiftKey: true });
    expect(c.toggleLabels).toHaveBeenCalledOnce();
    press({ key: 'R', shiftKey: true });
    expect(c.editRegions).toHaveBeenCalledOnce();
  });

  it('zooms with = - 0 and 1', () => {
    const c = fakeController();
    dispose = registerEditorKeys(c);
    press({ key: '=' });
    press({ key: '-' });
    press({ key: '0' });
    press({ key: '1' });
    expect(c.zoomIn).toHaveBeenCalledOnce();
    expect(c.zoomOut).toHaveBeenCalledOnce();
    expect(c.zoomFit).toHaveBeenCalledOnce();
    expect(c.zoomActual).toHaveBeenCalledOnce();
  });

  it('nudges one pixel, ten with shift, and resizes with alt+shift', () => {
    const c = fakeController();
    dispose = registerEditorKeys(c);
    press({ key: 'ArrowLeft' });
    expect(c.nudge).toHaveBeenLastCalledWith(-1, 0);
    press({ key: 'ArrowDown', shiftKey: true });
    expect(c.nudgeFar).toHaveBeenLastCalledWith(0, 1);
    press({ key: 'ArrowRight', altKey: true, shiftKey: true });
    expect(c.resizeCorner).toHaveBeenLastCalledWith(1, 0);
  });

  it('delete and backspace delete the selection', () => {
    const c = fakeController();
    dispose = registerEditorKeys(c);
    press({ key: 'Delete' });
    press({ key: 'Backspace' });
    expect(c.deleteSelected).toHaveBeenCalledTimes(2);
  });

  it('leaves arrows, delete and Tab to the browser when nothing is selected', () => {
    const c = fakeController({ selected: null, objects: [] });
    dispose = registerEditorKeys(c);
    expect(press({ key: 'ArrowUp' }).defaultPrevented).toBe(false);
    expect(press({ key: 'Delete' }).defaultPrevented).toBe(false);
    expect(press({ key: 'Tab' }).defaultPrevented).toBe(false);
    expect(c.nudge).not.toHaveBeenCalled();
  });

  it('Tab and Shift+Tab cycle the objects', () => {
    const c = fakeController();
    dispose = registerEditorKeys(c);
    expect(press({ key: 'Tab' }).defaultPrevented).toBe(true);
    press({ key: 'Tab', shiftKey: true });
    expect(c.selectNext).toHaveBeenCalledOnce();
    expect(c.selectPrev).toHaveBeenCalledOnce();
  });

  it('follows a rebind with no re-registration', () => {
    const c = fakeController();
    dispose = registerEditorKeys(c);
    keymapStore.setDocument(withKeys({ 'editor.tool_draw': ['x'] }), 'served');
    press({ key: 'b' });
    expect(c.setTool).not.toHaveBeenCalled();
    press({ key: 'x' });
    expect(c.setTool).toHaveBeenCalledWith('draw');
  });

  it('a class hotkey sets the selected object class through the drop bus', async () => {
    const c = fakeController();
    dispose = registerEditorKeys(c);
    const cls = { id: 7, name: 'widget' } as RegistryClass;
    await dropOnClassStore.dispatch(cls, []);
    expect(c.setClass).toHaveBeenCalledWith(cls);
  });

  it('takes the editor scope and gives it back', () => {
    keyboardStore.setScope('cluster');
    const off = registerEditorKeys(fakeController());
    expect(keyboardStore.scope).toBe(EDITOR_SCOPE);
    off();
    expect(keyboardStore.scope).toBe('cluster');
    expect(dropOnClassStore.handler).toBeNull();
  });

  it('registers nothing on window itself', () => {
    const add = vi.spyOn(window, 'addEventListener');
    dispose = registerEditorKeys(fakeController());
    expect(add.mock.calls.filter(([t]) => t === 'keydown')).toEqual([]);
    add.mockRestore();
  });
});
