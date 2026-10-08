import { describe, expect, it } from 'vitest';
import { formatShortcutKey } from './keyboardDisplay';

describe('formatShortcutKey', () => {
  it('renders arrow keys as glyphs, not the literal combo string', () => {
    expect(formatShortcutKey('arrowright')).toBe('→');
    expect(formatShortcutKey('arrowleft')).toBe('←');
  });

  it('capitalizes modifiers and single letters', () => {
    expect(formatShortcutKey('shift+n')).toBe('Shift+N');
    expect(formatShortcutKey('ctrl+shift+enter')).toBe('Ctrl+Shift+Enter');
  });

  it('title-cases multi-letter key names it does not special-case', () => {
    expect(formatShortcutKey('escape')).toBe('Esc');
  });
});
