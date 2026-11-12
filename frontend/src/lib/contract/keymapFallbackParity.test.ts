/**
 * `FALLBACK_KEYMAP` mirrors the served vocabulary (`contracts/json/keymap_actions.json`):
 * the same contexts and the same actions with the same default keys. A client on the fallback therefore binds exactly what a served keymap would.
 */
import { describe, expect, it } from 'vitest';
import served from '$contracts/json/keymap_actions.json';
import { FALLBACK_KEYMAP } from '$lib/keymapFallback';

type Row = [id: string, context: string, keys: string[]];

describe('FALLBACK_KEYMAP parity with the served vocabulary', () => {
  it('has the same contexts, with the same includes', () => {
    expect(FALLBACK_KEYMAP.contexts.map((c) => [c.id, c.includes])).toEqual(
      served.contexts.map((c) => [c.id, c.includes]),
    );
  });

  it('serves the workspace actions with their default keys', () => {
    const keys = Object.fromEntries(
      FALLBACK_KEYMAP.actions.map((a) => [a.id, a.default]),
    );
    expect(keys['workspace.escape']).toEqual(['escape']);
    expect(keys['workspace.view_next']).toEqual([']']);
    expect(keys['cluster.confirm']).toEqual(['shift+enter']);
    expect(keys['cluster.select_all']).toEqual(['a', 'ctrl+a', 'meta+a']);
  });

  it('serves the editor actions with their default keys', () => {
    const keys = Object.fromEntries(
      FALLBACK_KEYMAP.actions.map((a) => [a.id, a.default]),
    );
    expect(keys['editor.tool_draw']).toEqual(['b']);
    expect(keys['editor.cancel']).toEqual(['escape']);
    expect(keys['editor.undo']).toEqual(['ctrl+z', 'meta+z']);
  });

  it('has the same actions, contexts and default keys', () => {
    // The fallback lists `~` first so the overlay prints "~"; the served default omits it.
    const OVERLAY = 'global.shortcuts_overlay';
    const pick = (a: { id: string; context: string; default: string[] }): Row => [
      a.id,
      a.context,
      [...a.default].sort(),
    ];
    const byId = (x: Row, y: Row) => x[0].localeCompare(y[0]);
    const rows = (actions: Array<{ id: string; context: string; default: string[] }>) =>
      actions
        .filter((a) => a.id !== OVERLAY)
        .map(pick)
        .sort(byId);
    expect(rows(FALLBACK_KEYMAP.actions)).toEqual(rows(served.actions));
  });
});
