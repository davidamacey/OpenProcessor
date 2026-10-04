/**
 * Cards and tiles are never wrapped in `content-visibility: auto`. WebKit can
 * leave such an element unpainted: on Safari 26.6 (a Mac on the LAN) the
 * cluster and crop grids showed no images or cards although every thumbnail
 * request returned 200 with its bytes, and the same page renders fully in
 * Chromium. The grids are paged by an infinite scroller and their images are
 * `loading="lazy"`, so off-screen culling bought little and risked a blank page.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SRC = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) return sourceFiles(full);
    return /\.(ts|svelte|css)$/.test(name) && !name.endsWith('.test.ts') ? [full] : [];
  });
}

const CULLING = /content-visibility\s*:\s*auto|contain-intrinsic-size/;

describe('off-screen culling', () => {
  it('the pattern catches the inline style that was removed (negative control)', () => {
    expect(
      CULLING.test('style="content-visibility:auto;contain-intrinsic-size:auto 260px"'),
    ).toBe(true);
    expect(CULLING.test('class="aspect-square w-full"')).toBe(false);
  });

  it('no source file uses content-visibility:auto', () => {
    const offenders = sourceFiles(SRC).filter((f) =>
      CULLING.test(readFileSync(f, 'utf8')),
    );
    expect(offenders.map((f) => path.relative(SRC, f))).toEqual([]);
  });
});
