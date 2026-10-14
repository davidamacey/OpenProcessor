/**
 * Source text of the `/review` page for the source-scan tests that pin its
 * wiring. The page is split by concern: the route file wires controllers
 * from `lib/review/` to components from `lib/components/review/`, so a scan
 * reads the one file that now holds the code it pins (`reviewSource`), and a
 * "nowhere in the page" scan reads all of them (`reviewPageSource`).
 */

import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const srcRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

const PAGE_FILE = 'routes/p/[project]/review/+page.svelte';
const PAGE_DIRS = ['lib/review', 'lib/components/review'];

/** One file of the review page, by its path under `src/`. */
export function reviewSource(rel: string): string {
  return readFileSync(path.join(srcRoot, rel), 'utf-8');
}

/** Paths under `src/` of every non-test file the review page is made of. */
function reviewPageFiles(): string[] {
  const files = [PAGE_FILE];
  for (const dir of PAGE_DIRS) {
    for (const name of readdirSync(path.join(srcRoot, dir)).sort()) {
      if (/\.(svelte|ts)$/.test(name) && !/\.test\.ts$/.test(name)) {
        files.push(`${dir}/${name}`);
      }
    }
  }
  return files;
}

/** Every file of the review page, concatenated. */
export function reviewPageSource(): string {
  return reviewPageFiles()
    .map((rel) => reviewSource(rel))
    .join('\n');
}
