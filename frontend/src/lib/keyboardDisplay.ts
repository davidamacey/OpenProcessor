/**
 * Human-readable rendering of a normalized keyboard combo string
 * (`keyboard.svelte.ts`'s `normalize()` format, e.g. "shift+arrowright").
 *
 * p8 (2026-09-24 interactive pass): the shortcut overlay printed the raw
 * combo, so arrow-key bindings read as the literal string "arrowright".
 */
const KEY_NAMES: Record<string, string> = {
  arrowright: '→',
  arrowleft: '←',
  arrowup: '↑',
  arrowdown: '↓',
  escape: 'Esc',
  enter: 'Enter',
  ' ': 'Space',
  space: 'Space',
};

function formatPart(part: string): string {
  if (part in KEY_NAMES) return KEY_NAMES[part];
  if (part === 'ctrl') return 'Ctrl';
  if (part === 'meta') return 'Cmd';
  if (part === 'alt') return 'Alt';
  if (part === 'shift') return 'Shift';
  if (part.length === 1) return part.toUpperCase();
  return part.charAt(0).toUpperCase() + part.slice(1);
}

export function formatShortcutKey(combo: string): string {
  return combo.split('+').map(formatPart).join('+');
}
