/**
 * A column header must have text. An empty `<th>` is read as nothing by a
 * screen reader (axe `empty-table-header`); an actions or selection column
 * carries a visually hidden label instead.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SRC = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const EMPTY_TH = /<th\b[^>]*>\s*<\/th>/g;

function svelteFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) return svelteFiles(full);
    return name.endsWith('.svelte') ? [full] : [];
  });
}

describe('table headers', () => {
  it('the pattern catches an empty header (negative control)', () => {
    expect('<th class="w-8"></th>'.match(EMPTY_TH)).toHaveLength(1);
    expect('<th><span class="sr-only">Actions</span></th>'.match(EMPTY_TH)).toBeNull();
  });

  it('no <th> is empty', () => {
    const offenders = svelteFiles(SRC).flatMap((f) =>
      (readFileSync(f, 'utf-8').match(EMPTY_TH) ?? []).map(
        (m) => `${path.relative(SRC, f)}: ${m}`,
      ),
    );
    expect(offenders).toEqual([]);
  });
});
