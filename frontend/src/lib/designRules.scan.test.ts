/**
 * The design rules, enforced over the product source (every non-test .svelte and .ts under
 * src, comments stripped): no emoji, no Unicode glyph used as an icon (icons are Lucide SVGs
 * through `ui/Icon.svelte`), no pixel-sized `text-[Npx]`, no raw palette colour classes (colours
 * are the tokens in app.css `@theme`). Prose arrows (`→`) and the multiplication sign in a
 * dimension or a count ("1280×720", "models × datasets") are allowed; a lone `×` as a close
 * button is not. Each exception below carries its reason.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const SRC = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) return sourceFiles(full);
    if (/\.test\.ts$/.test(name) || /\.(svelte|ts)$/.test(name) === false) return [];
    return [full];
  });
}

/** Block and line comments and HTML comments out; a `//` only starts a comment after
 *  whitespace or at the line start, so URLs survive. */
export function stripComments(src: string): string {
  return src
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|\s)\/\/.*$/gm, '$1');
}

const FILES = sourceFiles(SRC)
  .map((f) => ({
    rel: path.relative(SRC, f),
    code: stripComments(readFileSync(f, 'utf-8')),
  }))
  .filter(({ rel }) => !rel.startsWith(path.join('lib', 'test') + path.sep));

const EMOJI = /\p{Extended_Pictographic}/u;
// Arrows (except the prose `→`), technical symbols, enclosed alphanumerics, geometric shapes,
// miscellaneous symbols, dingbats and the supplemental arrows block, plus the single and double
// angle quotes (‹ › « ») that stand in for chevrons.
const GLYPH_ICON = /[«»‹›←-↑↓-⇿⌀-⏿①-⓿■-◿☀-➿⬀-⯿]/u;
// A close button spelled "×": alone on its line, between tags, or trailing a label.
const CLOSE_GLYPH = /(^|>)\s*×\s*(<|$)|\S\s×\s*$|['"]×['"]/m;
const PIXEL_TEXT = /\btext-\[[\d.]+(px|rem|em)\]/;
const RAW_COLOUR =
  /(?<![\w-])(?:[a-z0-9-]+:)*(?:bg|text|border(?:-[trblxyse])?|ring|divide|outline|fill|stroke|from|to|via|accent|placeholder|decoration|shadow)-(?:zinc|slate|gray|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-\d{2,3}(?![\w-])/;

/** file (relative to src) -> why it may break a rule. */
const ALLOWED: Record<string, { rule: 'glyph' | 'colour'; why: string }> = {
  'lib/keyboardDisplay.ts': {
    rule: 'glyph',
    why: 'keycap legends (arrows, shift, return, backspace) are text the keymap displays, not icons',
  },
  'lib/annotations/config/allowLists.ts': {
    rule: 'colour',
    why: 'the sub-box ring presets are the operator-facing profile contract (examples/annotation-profiles)',
  },
  'lib/annotations/servedRegionSlot.ts': {
    rule: 'colour',
    why: 'the served-slot default ring is the same contract as the ring presets',
  },
  'lib/components/CropCard.svelte': {
    rule: 'colour',
    why: 'the fallback ring when no slot is served is the same contract as the ring presets',
  },
};

function offenders(
  test: (code: string) => boolean,
  rule: 'glyph' | 'colour' | 'none',
): string[] {
  return FILES.filter(({ rel, code }) => {
    const allowed = ALLOWED[path.sep === '/' ? rel : rel.replaceAll(path.sep, '/')];
    return test(code) && !(allowed && allowed.rule === rule);
  }).map(({ rel }) => rel);
}

describe('the patterns catch each spelling (negative controls)', () => {
  it('emoji and glyph icons', () => {
    expect(EMOJI.test('<span>✋</span>')).toBe(true);
    expect(EMOJI.test('<span>🔒 keys</span>')).toBe(true);
    for (const g of [
      '✓',
      '✎',
      '✗',
      'ⓘ',
      '▼',
      '▶',
      '▭',
      '●',
      '○',
      '◤',
      '↗',
      '↳',
      '↑',
      '←',
      '⇧',
      '‹',
      '›',
      '«',
      '»',
    ])
      expect(GLYPH_ICON.test(`<b>${g}</b>`), g).toBe(true);
    expect(GLYPH_ICON.test('Settings → Models')).toBe(false);
  });
  it('close glyphs, but not a dimension or a count', () => {
    expect(CLOSE_GLYPH.test('>\n  ×\n</button>')).toBe(true);
    expect(CLOSE_GLYPH.test('<span>×</span>')).toBe(true);
    expect(CLOSE_GLYPH.test('{chip.label} ×')).toBe(true);
    expect(CLOSE_GLYPH.test("label: '×'")).toBe(true);
    expect(CLOSE_GLYPH.test('<dd>{w}×{h}</dd>')).toBe(false);
    expect(CLOSE_GLYPH.test('{n} model × {m} dataset')).toBe(false);
  });
  it('pixel text sizes and raw colours', () => {
    expect(PIXEL_TEXT.test('class="text-[10px] p-1"')).toBe(true);
    expect(PIXEL_TEXT.test('class="text-caption"')).toBe(false);
    expect(RAW_COLOUR.test('class="hover:bg-zinc-800"')).toBe(true);
    expect(RAW_COLOUR.test('class="border-t-red-500/40"')).toBe(true);
    expect(RAW_COLOUR.test('class="bg-surface-2 text-danger-text border-line"')).toBe(
      false,
    );
    expect(RAW_COLOUR.test('class="bg-black/60 text-white"')).toBe(false);
  });
  it('comments are not scanned, URLs are kept', () => {
    expect(stripComments('a // ✓ done\nb')).not.toContain('✓');
    expect(stripComments('<!-- ✓ --><p>x</p>')).toBe('<p>x</p>');
    expect(stripComments("const u = 'http://x/✓'")).toContain('✓');
  });
});

describe('product source', () => {
  it('has no emoji', () => {
    expect(offenders((c) => EMOJI.test(c), 'none')).toEqual([]);
  });

  it('draws every icon as an SVG: no Unicode glyph icons or text close buttons', () => {
    expect(offenders((c) => GLYPH_ICON.test(c) || CLOSE_GLYPH.test(c), 'glyph')).toEqual(
      [],
    );
  });

  it('sizes text from the type scale, never text-[Npx]', () => {
    expect(offenders((c) => PIXEL_TEXT.test(c), 'none')).toEqual([]);
  });

  it('colours from the tokens, never a raw palette class', () => {
    expect(offenders((c) => RAW_COLOUR.test(c), 'colour')).toEqual([]);
  });

  it('keeps every exception to a file that still needs it', () => {
    for (const [file, { rule }] of Object.entries(ALLOWED)) {
      const hit = FILES.find((f) => f.rel.replaceAll(path.sep, '/') === file);
      expect(hit, `${file} no longer exists`).toBeDefined();
      const broken =
        rule === 'glyph'
          ? GLYPH_ICON.test(hit!.code) || CLOSE_GLYPH.test(hit!.code)
          : RAW_COLOUR.test(hit!.code);
      expect(
        broken,
        `${file} no longer breaks the ${rule} rule: drop it from ALLOWED`,
      ).toBe(true);
    }
  });
});
