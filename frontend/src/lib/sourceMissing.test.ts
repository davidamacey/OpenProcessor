import { describe, expect, it, vi } from 'vitest';
import { fetchSourceMissing, parseSourceMissing } from '$lib/sourceMissing';

const body = {
  detail: {
    error: 'source_missing',
    alias: 'archive',
    root_mounted: false,
    message: "source root 'archive' is not mounted or reachable",
  },
};

describe('sourceMissing', () => {
  it('parses the typed detail', () => {
    expect(parseSourceMissing(body)).toEqual({
      alias: 'archive',
      root_mounted: false,
      message: "source root 'archive' is not mounted or reachable",
    });
  });

  it('ignores other errors', () => {
    expect(parseSourceMissing({ detail: 'image missing' })).toBeNull();
    expect(parseSourceMissing({ detail: { error: 'unknown_box_id' } })).toBeNull();
  });

  it('reads the answer from a 404 response', async () => {
    const f = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify(body), { status: 404 }));
    expect((await fetchSourceMissing('/x', f as unknown as typeof fetch))?.alias).toBe(
      'archive',
    );
  });

  it('is null on a non-404 or a network failure', async () => {
    const ok = vi.fn().mockResolvedValue(new Response('{}', { status: 500 }));
    expect(await fetchSourceMissing('/x', ok as unknown as typeof fetch)).toBeNull();
    const bad = vi.fn().mockRejectedValue(new Error('net'));
    expect(await fetchSourceMissing('/x', bad as unknown as typeof fetch)).toBeNull();
  });
});
