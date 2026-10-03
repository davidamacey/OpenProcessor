import { describe, expect, it, vi } from 'vitest';
import {
  RELOAD_GUARD_KEY,
  RELOAD_WINDOW_MS,
  isChunkLoadError,
  mayAutoReload,
  recoverFromChunkError,
} from './chunkRecovery';

function memoryStorage(initial: Record<string, string> = {}) {
  const data = new Map(Object.entries(initial));
  return {
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => void data.set(k, v),
  };
}

describe('isChunkLoadError', () => {
  it.each([
    'Failed to fetch dynamically imported module: http://x/_app/immutable/nodes/3.js',
    'error loading dynamically imported module',
    'Importing a module script failed.',
    'Unable to preload CSS for /_app/a.css',
    'Failed to load module script: Expected a JavaScript module',
  ])('matches %s', (m) => expect(isChunkLoadError(new TypeError(m))).toBe(true));

  it('ignores unrelated errors and non-errors', () => {
    expect(isChunkLoadError(new Error('boom'))).toBe(false);
    expect(isChunkLoadError(null)).toBe(false);
    expect(
      isChunkLoadError({ message: 'Failed to fetch dynamically imported module' }),
    ).toBe(true);
  });
});

describe('mayAutoReload', () => {
  it('allows with no prior reload, blocks inside the window, allows after it', () => {
    expect(mayAutoReload(null, 1000)).toBe(true);
    expect(mayAutoReload(1000, 1000 + RELOAD_WINDOW_MS - 1)).toBe(false);
    expect(mayAutoReload(1000, 1000 + RELOAD_WINDOW_MS)).toBe(true);
  });
  it('treats a stamp in the future as recent', () => {
    expect(mayAutoReload(5000, 1000)).toBe(false);
  });
});

describe('recoverFromChunkError', () => {
  it('reloads once, not twice within the window, and again after it', () => {
    const storage = memoryStorage();
    const reload = vi.fn();
    expect(recoverFromChunkError(storage, 1000, reload)).toBe(true);
    expect(recoverFromChunkError(storage, 2000, reload)).toBe(false);
    expect(reload).toHaveBeenCalledTimes(1);
    expect(storage.getItem(RELOAD_GUARD_KEY)).toBe('1000');
    expect(recoverFromChunkError(storage, 1000 + RELOAD_WINDOW_MS, reload)).toBe(true);
    expect(reload).toHaveBeenCalledTimes(2);
  });

  it('fails closed when storage is unavailable or throws', () => {
    const reload = vi.fn();
    expect(recoverFromChunkError(null, 1000, reload)).toBe(false);
    const broken = {
      getItem: () => {
        throw new Error('denied');
      },
      setItem: () => {},
    };
    expect(recoverFromChunkError(broken, 1000, reload)).toBe(false);
    expect(reload).not.toHaveBeenCalled();
  });

  it('treats a corrupt stamp as no stamp', () => {
    const reload = vi.fn();
    expect(
      recoverFromChunkError(memoryStorage({ [RELOAD_GUARD_KEY]: 'abc' }), 1, reload),
    ).toBe(true);
  });
});
