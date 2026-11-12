import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { FALLBACK_KEYMAP } from '$lib/keymapFallback';
import { keymapStore } from '$stores/keymap.svelte';
import ToolStrip from './ToolStrip.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

beforeEach(() => keymapStore.resetToFallback());
afterEach(() => {
  keymapStore.resetToFallback();
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function render(over: Record<string, unknown> = {}) {
  const calls = {
    ontool: vi.fn(),
    onzoomin: vi.fn(),
    onzoomout: vi.fn(),
    onfit: vi.fn(),
    onactual: vi.fn(),
    onundo: vi.fn(),
    onredo: vi.fn(),
    onlabels: vi.fn(),
  };
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ToolStrip, {
    target,
    props: {
      tool: 'select',
      sticky: false,
      labels: true,
      zoomPercent: 100,
      canUndo: false,
      canRedo: false,
      undoLabel: null,
      redoLabel: null,
      ...calls,
      ...over,
    },
  });
  flushSync();
  return calls;
}

const btn = (id: string) =>
  target.querySelector(`[data-action="${id}"]`) as HTMLButtonElement;

describe('ToolStrip', () => {
  it('marks the active tool', () => {
    render({ tool: 'draw' });
    expect(btn('editor.tool_draw').getAttribute('aria-pressed')).toBe('true');
    expect(btn('editor.tool_select').getAttribute('aria-pressed')).toBe('false');
  });

  it('tells the sticky draw tool from the single one', () => {
    render({ tool: 'draw', sticky: true });
    expect(btn('editor.tool_draw_sticky').getAttribute('aria-pressed')).toBe('true');
    expect(btn('editor.tool_draw').getAttribute('aria-pressed')).toBe('false');
  });

  it('its buttons call back', () => {
    const c = render({ canUndo: true, canRedo: true });
    btn('editor.tool_pan').click();
    expect(c.ontool).toHaveBeenCalledWith('pan');
    btn('editor.tool_draw_sticky').click();
    expect(c.ontool).toHaveBeenCalledWith('draw', true);
    btn('editor.zoom_in').click();
    btn('editor.zoom_out').click();
    btn('editor.zoom_fit').click();
    btn('editor.zoom_actual').click();
    btn('editor.undo').click();
    btn('editor.redo').click();
    btn('editor.labels').click();
    expect(c.onzoomin).toHaveBeenCalledOnce();
    expect(c.onzoomout).toHaveBeenCalledOnce();
    expect(c.onfit).toHaveBeenCalledOnce();
    expect(c.onactual).toHaveBeenCalledOnce();
    expect(c.onundo).toHaveBeenCalledOnce();
    expect(c.onredo).toHaveBeenCalledOnce();
    expect(c.onlabels).toHaveBeenCalledOnce();
  });

  it('disables undo and redo until there is something to undo or redo', () => {
    render();
    expect(btn('editor.undo').disabled).toBe(true);
    expect(btn('editor.redo').disabled).toBe(true);
  });

  it('names what undo would undo, with its key', () => {
    render({ canUndo: true, undoLabel: 'Draw widget' });
    expect(btn('editor.undo').title).toBe(
      `Undo: Draw widget (${keymapStore.glyph('editor.undo')})`,
    );
  });

  it('shows the zoom as a percentage', () => {
    render({ zoomPercent: 62.4 });
    expect(btn('editor.zoom_actual').textContent?.trim()).toBe('62%');
  });

  it('prints the key the keymap serves, so a rebind changes the hint', () => {
    keymapStore.setDocument(
      {
        ...FALLBACK_KEYMAP,
        actions: FALLBACK_KEYMAP.actions.map((a) =>
          a.id === 'editor.tool_draw' ? { ...a, keys: ['x'] } : a,
        ),
      },
      'served',
    );
    render();
    expect(btn('editor.tool_draw').title).toBe('Draw a box (X)');
  });
});
