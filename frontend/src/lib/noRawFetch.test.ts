import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

/**
 * Every network call goes through the API modules (`$lib/api*.ts`), which
 * carry the project scope, timeouts and error mapping. A raw `fetch(` in a
 * page or component bypasses all three.
 */
const srcRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) return sourceFiles(full);
    return /\.(ts|svelte)$/.test(name) && !/\.test\.ts$/.test(name) ? [full] : [];
  });
}

describe('raw fetch', () => {
  it('is used only by the API modules', () => {
    const offenders = sourceFiles(srcRoot)
      .filter((f) => !/[\\/]lib[\\/]api[^\\/]*\.ts$/.test(f))
      .filter((f) => /\bfetch\(/.test(readFileSync(f, 'utf-8')))
      .map((f) => path.relative(srcRoot, f));
    expect(offenders).toEqual([]);
  });
});
