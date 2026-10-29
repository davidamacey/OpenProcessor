/**
 * Dim text (`text-zinc-500` for hints and captions, `text-zinc-600` for the
 * dimmest labels) must stay legible on the surfaces it sits on. Tailwind's
 * stock zinc-500/600 measure 3.7:1 and 2.6:1 against the dark card surface,
 * under the WCAG AA 4.5:1 floor, so app.css lifts both in `@theme`.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const css = readFileSync(
  path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../app.css'),
  'utf-8',
);

function channel(v: number): number {
  const c = v / 255;
  return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function luminance(hex: string): number {
  const n = parseInt(hex.slice(1), 16);
  return (
    0.2126 * channel((n >> 16) & 255) +
    0.7152 * channel((n >> 8) & 255) +
    0.0722 * channel(n & 255)
  );
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi! + 0.05) / (lo! + 0.05);
}

function themeColor(name: string): string {
  const m = css.match(new RegExp(`--color-${name}:\\s*(#[0-9a-fA-F]{6})`));
  expect(m, `app.css must override --color-${name} in @theme`).not.toBeNull();
  return m![1]!;
}

const CARD_SURFACE = '#18181b'; // zinc-900, the lightest surface dim text sits on

describe('dim text contrast', () => {
  it.each(['zinc-500', 'zinc-600'])('%s reaches 4.5:1 on the card surface', (name) => {
    expect(contrast(themeColor(name), CARD_SURFACE)).toBeGreaterThanOrEqual(4.5);
  });
});

describe('filled button contrast', () => {
  it.each(['btn-primary-bg', 'btn-danger-bg'])('white on --%s reaches 4.5:1', (name) => {
    const m = css.match(new RegExp(`--${name}:\\s*(\\d+) (\\d+) (\\d+);`));
    expect(m, `app.css must define --${name}`).not.toBeNull();
    const hex =
      '#' +
      m!
        .slice(1, 4)
        .map((n) => Number(n).toString(16).padStart(2, '0'))
        .join('');
    expect(contrast('#ffffff', hex)).toBeGreaterThanOrEqual(4.5);
  });
});
