/**
 * The gallery renders the order the server served. A client sort, reverse or narrowing added
 * back fails here.
 */
import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const LIB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const FILES = [
  'api_images.ts',
  'images/imageQuery.ts',
  'images/imageGalleryController.svelte.ts',
];

const RERANK = /\.(sort|toSorted|reverse|toReversed)\(/;

describe('the image list is rendered in served order', () => {
  it('the rerank pattern catches the spellings (negative control)', () => {
    expect(RERANK.test('out.sort((a, b) => a.size - b.size)')).toBe(true);
    expect(RERANK.test('list.toSorted(cmp)')).toBe(true);
    expect(RERANK.test('cards.reverse()')).toBe(true);
    expect(RERANK.test('pager.items.slice(0, shown)')).toBe(false);
  });

  it.each(FILES)('%s does not sort or reverse the cards', (file) => {
    const full = path.join(LIB, file);
    expect(existsSync(full), `${file} exists`).toBe(true);
    const lines = readFileSync(full, 'utf-8').split('\n');
    const hits = lines.filter((l) => !l.trim().startsWith('//') && RERANK.test(l));
    expect(hits).toEqual([]);
  });
});
