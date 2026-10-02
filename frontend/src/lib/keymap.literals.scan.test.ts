/**
 * Ratchet for configurable keyboard shortcuts (docs/design/configurable-
 * keyboard-shortcuts-plan-2026-09-26.md §5.5, step K1): every key the UI
 * prints comes from `keymapStore` (`glyph`/`compactGlyph`), never a
 * literal — a literal goes stale the moment a binding changes.
 *
 * Fails on, outside comments: a `<kbd>` whose text is a literal, "Press X"
 * / "press X" naming a single key, and a "(X)" / "(Enter)" / "(Esc)" key
 * reference.
 *
 * `ALLOWED` is exact and per-file. The keymap's own data/formatting files
 * are the one place key names belong, and the §1.2 fixed form keys (the
 * class-picker's listbox hint) are platform keys, not shortcuts. Never add
 * an entry to make a red run green — read the key from the keymap.
 *
 * Mutation-check: put `<kbd>D</kbd>` back into a scratch copy of
 * `review/+page.svelte`'s hint strip and this test goes red.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { stripComments } from '$lib/testing/sourceScan';

const here = path.dirname(fileURLToPath(import.meta.url));
const srcRoot = path.resolve(here, '..');

const SKIP_FILES = new Set([
  // The keymap's own defaults and labels.
  'lib/keymapFallback.ts',
  // Combo -> display-name formatting.
  'lib/keyboardDisplay.ts',
]);

/** Exact substrings allowed in one file, each with its reason. */
const ALLOWED: Record<string, string[]> = {
  // §1.2: the class picker's listbox keys are fixed form keys.
  'routes/p/[project]/review/+page.svelte': [
    '<kbd>↑↓</kbd> navigate · <kbd>Enter</kbd> assign · <kbd>Esc</kbd> close',
  ],
};

const PATTERNS: Array<[string, RegExp]> = [
  ['literal <kbd>', /<kbd[^>]*>\s*[^\s<{][^<{]*<\/kbd\s*>/g],
  ['"press <key>"', /\b[Pp]ress (?:[A-Z]|Enter|Esc|Backspace|[/~←→↑↓⌫↵])(?![\w])/g],
  ['"(<key>)"', /\((?:[A-Z]|Enter|Esc|⇧.|[/~←→↑↓⌫↵])\)/g],
];

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = path.join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (/\.(svelte|ts)$/.test(name) && !/\.test\.ts$/.test(name)) out.push(p);
  }
  return out;
}

describe('no literal keys printed outside the keymap (K1 ratchet)', () => {
  it('every printed key reads the keymap', () => {
    const offenders: string[] = [];
    for (const file of walk(srcRoot)) {
      const rel = path.relative(srcRoot, file).split(path.sep).join('/');
      if (SKIP_FILES.has(rel)) continue;
      let text = stripComments(readFileSync(file, 'utf-8'));
      for (const allowed of ALLOWED[rel] ?? []) text = text.split(allowed).join(' ');
      for (const [what, re] of PATTERNS) {
        for (const m of text.matchAll(re)) offenders.push(`${rel}: ${what}: ${m[0]}`);
      }
    }
    expect(offenders).toEqual([]);
  });

  it('every allow-list entry still exists (no stale exemptions)', () => {
    for (const [rel, entries] of Object.entries(ALLOWED)) {
      const text = readFileSync(path.join(srcRoot, rel), 'utf-8');
      for (const e of entries) expect(text, `${rel}: ${e}`).toContain(e);
    }
  });
});
