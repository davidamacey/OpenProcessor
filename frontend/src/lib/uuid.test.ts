import { afterEach, describe, expect, it, vi } from 'vitest';
import { newUuid } from './uuid';

const V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

afterEach(() => vi.unstubAllGlobals());

describe('newUuid', () => {
  it('returns a v4 uuid', () => {
    expect(newUuid()).toMatch(V4);
  });

  it('still returns one where randomUUID does not exist (an HTTP page)', () => {
    vi.stubGlobal('crypto', {
      getRandomValues: (a: Uint8Array) => a.fill(255),
    });
    expect(newUuid()).toMatch(V4);
  });

  it('is different each time', () => {
    expect(newUuid()).not.toBe(newUuid());
  });
});
