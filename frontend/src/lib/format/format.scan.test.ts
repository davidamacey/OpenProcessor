/**
 * Duration, byte and time formatters have one home, `lib/format/`. A page or
 * controller that re-declares one drifts from the others (three panels once
 * printed the same ETA three different ways).
 */
import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SRC = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const FORMAT_DIR = path.join(SRC, 'lib', 'format');

const DECLARATION =
  /\b(?:function|const|let)\s+(formatDuration|fmtBytes|formatBytes|formatTime|fmtEta)\b/;

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) return sourceFiles(full);
    return /\.(ts|svelte)$/.test(name) && !name.endsWith('.test.ts') ? [full] : [];
  });
}

describe('formatters', () => {
  it('the pattern catches each declaration spelling (negative control)', () => {
    expect(DECLARATION.test('function formatDuration(s: number) {')).toBe(true);
    expect(DECLARATION.test('  function fmtEta(sec: number | null): string {')).toBe(
      true,
    );
    expect(DECLARATION.test('const fmtBytes = (n: number) => n;')).toBe(true);
    expect(DECLARATION.test('export function formatTimestamp(v: string) {')).toBe(false);
    expect(DECLARATION.test('const label = formatDuration(5);')).toBe(false);
  });

  it('are declared only under lib/format/', () => {
    const offenders = sourceFiles(SRC).flatMap((f) => {
      if (f.startsWith(FORMAT_DIR + path.sep)) return [];
      const rel = path.relative(SRC, f);
      return readFileSync(f, 'utf-8')
        .split('\n')
        .filter((line) => DECLARATION.test(line))
        .map((line) => `${rel}: ${line.trim()}`);
    });
    expect(offenders).toEqual([]);
  });
});
