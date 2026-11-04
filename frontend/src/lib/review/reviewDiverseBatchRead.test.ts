import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

/**
 * A diverse page hydrates its ids with the one batch read, not one
 * `GET /crops/{id}` per id (30 requests a page). Static scan, as the other
 * controller guards in this tree.
 */
const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(
  path.resolve(here, 'reviewDiverseController.svelte.ts'),
  'utf-8',
);

describe('diverse review page', () => {
  it('reads its items with getCropsByIds', () => {
    expect(src).toMatch(/getCropsByIds\(ids\)/);
  });

  it('does not fan out a getCrop per id', () => {
    expect(src).not.toMatch(/getCrop\(/);
    expect(src).not.toMatch(/Promise\.all/);
  });
});
