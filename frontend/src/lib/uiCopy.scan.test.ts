/**
 * Internal planning ids (backend-ask ids like "BA-2") are for docs and code
 * comments, never for text an operator reads. Scans the markup of every
 * .svelte file with <script>/<style> blocks and HTML comments stripped.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SRC = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TICKET_ID = /\bBA-\d+\b/;

function svelteFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) return svelteFiles(full);
    return name.endsWith('.svelte') ? [full] : [];
  });
}

function markupOnly(source: string): string {
  return source
    .replace(/<script[\s\S]*?<\/script>/g, '')
    .replace(/<style[\s\S]*?<\/style>/g, '')
    .replace(/<!--[\s\S]*?-->/g, '');
}

describe('rendered UI copy', () => {
  it('carries no internal planning ids', () => {
    const offenders = svelteFiles(SRC).flatMap((f) =>
      markupOnly(readFileSync(f, 'utf-8'))
        .split('\n')
        .filter((line) => TICKET_ID.test(line))
        .map((line) => `${path.relative(SRC, f)}: ${line.trim()}`),
    );
    expect(offenders).toEqual([]);
  });
});
