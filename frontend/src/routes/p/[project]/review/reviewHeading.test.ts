/**
 * The review workspace has no visible page title (the tab strip is its
 * header), so it carries a visually hidden h1 for screen-reader navigation.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

it('renders a visually hidden h1', () => {
  expect(src).toMatch(/<h1 class="sr-only">Review<\/h1>/);
});
