/**
 * The design rules, enforced over the product source (every non-test .svelte and .ts under
 * src, comments stripped): no emoji, no Unicode glyph used as an icon (icons are Lucide SVGs
 * through `ui/Icon.svelte`), no pixel-sized `text-[Npx]`, no raw palette colour classes (colours
 * are the tokens in app.css `@theme`). Prose arrows (`→`) and the multiplication sign in a
 * dimension or a count ("1280×720", "models × datasets") are allowed; a lone `×` as a close
 * button is not. Type follows the scale in app.css `@theme`: no Tailwind size utility
 * (`text-xs`..`text-9xl`), no `uppercase`, `font-mono` only in an allow-listed code editor
 * (ids, hashes and paths are written as `<code>`), and every page title goes through
 * `ui/PageHeader.svelte`. Each exception below carries its reason.
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
// The seven size tokens (text-caption..text-kpi) are the only sizes; Tailwind's own scale is not.
const TW_SIZE = /\btext-(xs|sm|base|lg|xl|[2-9]xl)\b/;
// `font-mono` is for code editors only; an id, hash or path is a `<code>` element.
const MONO_CLASS = /\bfont-mono\b/;
// A `<code>` that wraps a sentence. A class list or a string literal of the word is not a class use.
const CODE_PROSE = /<code[^>]*>[^<{]*\b[A-Za-z]{3,}\s+[A-Za-z]{3,}\s+[A-Za-z]{3,}/;
// The `uppercase` utility in a class list. The transform names in the slot config
// (`'uppercase'`, `uppercase: boolean`) are data, not a class.
const UPPERCASE = /(?<=\s|class=["'`{])uppercase(?![\w-]|\s*\??:)/;
// A page title is `PageHeader`'s h1; a workspace view keeps an `sr-only` one.
const RAW_H1 = /<h1(?![^>]*\bsr-only\b)[^>]*class=/;

/** file (relative to src) -> why it may break a rule. */
const ALLOWED: { file: string; rule: 'glyph' | 'colour' | 'mono' | 'h1'; why: string }[] =
  [
    {
      file: 'lib/components/ingest/SourceForm.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits an S3 bucket, key prefix, folder path and cron expression',
    },
    {
      file: 'lib/components/ingest/BulkIngestPanel.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits a server path',
    },
    {
      file: 'lib/components/ingest/IngestBatchPanel.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits a list of server paths',
    },
    {
      file: 'lib/components/combine/CombineSourcesStep.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits a project slug',
    },
    {
      file: 'lib/components/profiles/ProfileFieldEditor.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits a JSON value',
    },
    {
      file: 'lib/components/profiles/ProfileTestPanel.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits crop ids',
    },
    {
      file: 'lib/components/bakeoff/ModelPicker.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits a class-name mapping as JSON',
    },
    {
      file: 'lib/components/AugmentationPanel.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits a class:count list',
    },
    {
      file: 'lib/components/LogTail.svelte',
      rule: 'mono',
      why: 'a code editor: this file is a log viewer: a code view of raw lines',
    },
    {
      file: 'lib/components/projects/DeleteProjectDialog.svelte',
      rule: 'mono',
      why: 'a code editor: this file retypes a project slug to confirm',
    },
    {
      file: 'lib/components/projects/CreateProjectDialog.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits a project slug',
    },
    {
      file: 'lib/components/PromoteModal.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits a Triton model name',
    },
    {
      file: 'lib/components/presets/PresetNewDialog.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits a preset name and its JSON body',
    },
    {
      file: 'lib/components/config/ConfigCloneDialog.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits a config document name',
    },
    {
      file: 'lib/components/packs/PackFieldEditor.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits regex patterns, mapping keys and a JSON list',
    },
    {
      file: 'lib/components/packs/PackTestPanel.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits crop ids',
    },
    {
      file: 'lib/components/openVocab/OpenVocabTestPanel.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits a crop id',
    },
    {
      file: 'routes/p/[project]/settings/models/new-endpoint/+page.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits an endpoint name',
    },
    {
      file: 'routes/p/[project]/settings/presets/[kind]/[name]/+page.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits a preset body as JSON',
    },
    {
      file: 'routes/p/[project]/settings/open-vocab/+page.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits an open-vocabulary set name',
    },
    {
      file: 'lib/components/datasets/ImportWizardPanel.svelte',
      rule: 'mono',
      why: 'a code editor: this file edits a server path',
    },
    {
      file: 'lib/components/clusters/ClustersToolbar.svelte',
      rule: 'h1',
      why: 'the title moves to ui/PageHeader or an sr-only h1 when lane C restructures this view (12f section 4)',
    },
    {
      file: 'lib/components/editor/EditorHeader.svelte',
      rule: 'h1',
      why: 'the title moves to ui/PageHeader or an sr-only h1 when lane C restructures this view (12f section 4)',
    },
    {
      file: 'lib/components/review/ReviewHeader.svelte',
      rule: 'h1',
      why: 'the title moves to ui/PageHeader or an sr-only h1 when lane C restructures this view (12f section 4)',
    },
    {
      file: 'lib/components/shell/ErrorPage.svelte',
      rule: 'h1',
      why: 'a full-page error state, not a titled page; it has no PageHeader slots for an action row',
    },
    {
      file: 'lib/workspace/views/ClusterScope.svelte',
      rule: 'h1',
      why: 'the title moves to ui/PageHeader or an sr-only h1 when lane C restructures this view (12f section 4)',
    },
    {
      file: 'lib/keyboardDisplay.ts',
      rule: 'glyph',
      why: 'keycap legends (arrows, shift, return, backspace) are text the keymap displays, not icons',
    },
    {
      file: 'lib/annotations/config/allowLists.ts',
      rule: 'colour',
      why: 'the sub-box ring presets are the operator-facing profile contract (examples/annotation-profiles)',
    },
    {
      file: 'lib/annotations/servedRegionSlot.ts',
      rule: 'colour',
      why: 'the served-slot default ring is the same contract as the ring presets',
    },
    {
      file: 'lib/components/CropCard.svelte',
      rule: 'colour',
      why: 'the fallback ring when no slot is served is the same contract as the ring presets',
    },
  ];

function offenders(
  test: (code: string) => boolean,
  rule: 'glyph' | 'colour' | 'mono' | 'h1' | 'none',
): string[] {
  return FILES.filter(({ rel, code }) => {
    const file = rel.replaceAll(path.sep, '/');
    return test(code) && !ALLOWED.some((a) => a.file === file && a.rule === rule);
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
  it('Tailwind size utilities, but not the scale tokens', () => {
    for (const c of [
      'text-xs',
      'text-sm',
      'text-base',
      'text-lg',
      'text-xl',
      'text-2xl',
      'text-9xl',
    ])
      expect(TW_SIZE.test(`class="${c} p-1"`), c).toBe(true);
    expect(TW_SIZE.test('class="md:text-sm"')).toBe(true);
    for (const c of [
      'text-caption',
      'text-meta',
      'text-control',
      'text-body',
      'text-title',
      'text-h1',
      'text-kpi',
      'text-fg-1',
      'text-center',
    ])
      expect(TW_SIZE.test(`class="${c}"`), c).toBe(false);
  });
  it('font-mono, however it sits in a class list', () => {
    expect(MONO_CLASS.test('class="font-mono text-meta"')).toBe(true);
    expect(MONO_CLASS.test('class="text-meta font-mono"')).toBe(true);
    expect(MONO_CLASS.test('class="tabular-nums"')).toBe(false);
  });
  it('a <code> that wraps a sentence, but not an id or a path', () => {
    expect(CODE_PROSE.test('<code>the whole sentence here</code>')).toBe(true);
    expect(CODE_PROSE.test('<code class="x">items matching the filter</code>')).toBe(
      true,
    );
    expect(CODE_PROSE.test('<code>{item.id}</code>')).toBe(false);
    expect(CODE_PROSE.test('<code>/data/images/a.jpg</code>')).toBe(false);
    expect(CODE_PROSE.test('<code>OP_CURATION_PROJECT</code>')).toBe(false);
  });
  it('the uppercase utility, but not the slot transform data', () => {
    expect(UPPERCASE.test('class="uppercase tracking-wide"')).toBe(true);
    expect(UPPERCASE.test('class="text-meta uppercase"')).toBe(true);
    expect(UPPERCASE.test('class="text-meta uppercase text-fg-3"')).toBe(true);
    expect(UPPERCASE.test("transform?: 'none' | 'uppercase' | 'lowercase'")).toBe(false);
    expect(UPPERCASE.test('  uppercase: boolean;')).toBe(false);
    expect(UPPERCASE.test("=== 'uppercase'")).toBe(false);
    expect(UPPERCASE.test('class="normal-case"')).toBe(false);
  });
  it('a page h1 that bypasses PageHeader, but not an sr-only one', () => {
    expect(RAW_H1.test('<h1 class="text-h1 font-semibold">Title</h1>')).toBe(true);
    expect(RAW_H1.test('<h1 data-x="y" class="text-title">Title</h1>')).toBe(true);
    expect(RAW_H1.test('<h1 class="sr-only">Gallery</h1>')).toBe(false);
    expect(RAW_H1.test('<h1>Bare</h1>')).toBe(false);
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

  it('sizes text only with the seven scale tokens, never a Tailwind size utility', () => {
    expect(offenders((c) => TW_SIZE.test(c), 'none')).toEqual([]);
  });

  it('keeps mono for allow-listed code editors; ids and paths are <code>', () => {
    expect(offenders((c) => MONO_CLASS.test(c), 'mono')).toEqual([]);
  });

  it('never wraps a sentence in <code>', () => {
    expect(offenders((c) => CODE_PROSE.test(c), 'none')).toEqual([]);
  });

  it('writes sentence case, never the uppercase utility', () => {
    expect(offenders((c) => UPPERCASE.test(c), 'none')).toEqual([]);
  });

  it('gives every page title through PageHeader (a view keeps an sr-only h1)', () => {
    expect(
      offenders((c) => RAW_H1.test(c), 'h1').filter(
        (f) => f !== path.join('lib', 'components', 'ui', 'PageHeader.svelte'),
      ),
    ).toEqual([]);
  });

  it('keeps every exception to a file that still needs it', () => {
    const TESTS = {
      glyph: (c: string) => GLYPH_ICON.test(c) || CLOSE_GLYPH.test(c),
      colour: (c: string) => RAW_COLOUR.test(c),
      mono: (c: string) => MONO_CLASS.test(c),
      h1: (c: string) => RAW_H1.test(c),
    };
    for (const { file, rule } of ALLOWED) {
      const hit = FILES.find((f) => f.rel.replaceAll(path.sep, '/') === file);
      expect(hit, `${file} no longer exists`).toBeDefined();
      const broken = TESTS[rule](hit!.code);
      expect(
        broken,
        `${file} no longer breaks the ${rule} rule: drop it from ALLOWED`,
      ).toBe(true);
    }
  });
});
