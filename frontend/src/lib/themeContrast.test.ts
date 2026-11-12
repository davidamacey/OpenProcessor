/**
 * The design tokens in app.css `@theme` must keep text legible on the surfaces it sits on:
 * fg-2 and fg-3 reach WCAG AA (4.5:1) on every card-level surface, and the filled accent and
 * danger backgrounds carry white label text at 4.5:1.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const css = readFileSync(
  path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../app.css'),
  'utf-8',
);

type Rgb = [number, number, number];

function channel(v: number): number {
  const c = v / 255;
  return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function luminance([r, g, b]: Rgb): number {
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

function contrast(a: Rgb, b: Rgb): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi! + 0.05) / (lo! + 0.05);
}

function token(name: string): Rgb {
  const m = css.match(new RegExp(`--color-${name}:\\s*rgb\\((\\d+) (\\d+) (\\d+)\\)`));
  expect(m, `app.css @theme must define --color-${name} as rgb(r g b)`).not.toBeNull();
  return [Number(m![1]), Number(m![2]), Number(m![3])];
}

describe('text contrast on the surfaces', () => {
  const surfaces = ['surface-0', 'surface-1', 'surface-2', 'surface-3'];
  it.each(['fg-1', 'fg-2', 'fg-3'])('%s reaches 4.5:1 on every surface', (fg) => {
    for (const s of surfaces) {
      expect(contrast(token(fg), token(s)), `${fg} on ${s}`).toBeGreaterThanOrEqual(4.5);
    }
  });
});

describe('semantic text and the accent', () => {
  it.each(['accent', 'success', 'warn', 'danger', 'info', 'holdout'])(
    '%s text reaches 4.5:1 on the card surface',
    (name) => {
      expect(contrast(token(name), token('surface-1'))).toBeGreaterThanOrEqual(4.5);
    },
  );
});

describe('filled button contrast', () => {
  it.each(['accent-fill', 'danger-fill'])('white on --color-%s reaches 4.5:1', (name) => {
    expect(contrast([255, 255, 255], token(name))).toBeGreaterThanOrEqual(4.5);
  });
});
