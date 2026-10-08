import { describe, expect, it } from 'vitest';
import type { IngestFile } from './fileSource';
import { chunkForLookup, isOversize, planChunks } from './uploadPlanner';

function mkFile(id: string, size: number): IngestFile {
  return { id, relPath: id, file: new File(['x'.repeat(size)], id), size };
}

describe('planChunks', () => {
  it('respects the image-count cap', () => {
    const files = Array.from({ length: 5 }, (_, i) => mkFile(`f${i}`, 10));
    const chunks = planChunks(files, { maxImages: 2, maxBytes: 1_000_000 });
    expect(chunks.map((c) => c.length)).toEqual([2, 2, 1]);
  });

  it('respects the byte cap', () => {
    const files = [mkFile('a', 40), mkFile('b', 40), mkFile('c', 40)];
    const chunks = planChunks(files, { maxImages: 100, maxBytes: 90 });
    expect(chunks.map((c) => c.length)).toEqual([2, 1]);
  });

  it('isolates and flags an oversize file, never blocking the rest', () => {
    const files = [mkFile('a', 10), mkFile('huge', 1000), mkFile('b', 10)];
    const chunks = planChunks(files, { maxImages: 100, maxBytes: 100 });
    expect(chunks).toHaveLength(3);
    expect(chunks[0]!.map((f) => f.id)).toEqual(['a']);
    expect(chunks[1]!.map((f) => f.id)).toEqual(['huge']);
    expect(chunks[2]!.map((f) => f.id)).toEqual(['b']);
    const huge = chunks[1]![0]!;
    expect(isOversize(huge)).toBe(true);
  });

  it('preserves input order across chunk boundaries', () => {
    const files = Array.from({ length: 9 }, (_, i) => mkFile(`f${i}`, 10));
    const chunks = planChunks(files, { maxImages: 3, maxBytes: 1_000_000 });
    const flat = chunks.flatMap((c) => c.map((f) => f.id));
    expect(flat).toEqual(files.map((f) => f.id));
  });

  it('returns no chunks for an empty input', () => {
    expect(planChunks([], { maxImages: 10, maxBytes: 100 })).toEqual([]);
  });
});

describe('chunkForLookup', () => {
  it('chunks by the given max', () => {
    const ids = Array.from({ length: 25 }, (_, i) => `id${i}`);
    const chunks = chunkForLookup(ids, 10);
    expect(chunks.map((c) => c.length)).toEqual([10, 10, 5]);
    expect(chunks.flat()).toEqual(ids);
  });

  it('returns a single chunk when under the max', () => {
    expect(chunkForLookup(['a', 'b'], 10)).toEqual([['a', 'b']]);
  });

  it('returns no chunks for an empty input', () => {
    expect(chunkForLookup([], 10)).toEqual([]);
  });
});
