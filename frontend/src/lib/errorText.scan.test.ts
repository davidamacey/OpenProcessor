/**
 * User-visible error text goes through `apiErrorText` (api.ts), never a raw
 * `(e as Error).message` / `String(e)`: the raw message of an ApiError is
 * "API 503 /curation/projects/... - <detail>", which leaks the URL into toasts.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SRC = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const RAW_ERROR_TEXT = [
  /\bas Error\)\??\.message/,
  /\?\? String\((?:e|err|error|ex)\)/,
  /\b(\w+) instanceof Error \? \1\.message/,
  /\.detail \?\? \w+\?*\.message/,
];

// The helper itself, and the chunk-load recovery hook (no ApiError, no UI text).
const ALLOWED = new Set(['lib/api.ts', 'lib/chunkRecovery.ts']);

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) return sourceFiles(full);
    return /\.(ts|svelte)$/.test(name) && !name.endsWith('.test.ts') ? [full] : [];
  });
}

const hit = (line: string): boolean => RAW_ERROR_TEXT.some((re) => re.test(line));

describe('error text', () => {
  it('the patterns catch the raw spellings (negative control)', () => {
    expect(hit('toast((e as Error).message)')).toBe(true);
    expect(hit('x = (err as Error)?.message ?? String(err)')).toBe(true);
    expect(hit('e instanceof Error ? e.message : String(e)')).toBe(true);
    expect(hit("this.error = err?.detail ?? err?.message ?? 'x'")).toBe(true);
    expect(hit('toast(apiErrorText(e))')).toBe(false);
    expect(hit("if ((e as Error).name === 'AbortError') return;")).toBe(false);
  });

  it('is never composed from a raw Error message in product code', () => {
    const offenders = sourceFiles(SRC).flatMap((f) => {
      const rel = path.relative(SRC, f);
      if (ALLOWED.has(rel)) return [];
      return readFileSync(f, 'utf-8')
        .split('\n')
        .filter(hit)
        .map((line) => `${rel}: ${line.trim()}`);
    });
    expect(offenders).toEqual([]);
  });
});
