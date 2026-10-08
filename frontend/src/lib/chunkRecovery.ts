/**
 * Recovery from a failed module-chunk load (a flaky network, a browser
 * network-change abort, or a deploy that replaced the hashed chunks while a
 * tab was open). Pure decision logic only; `src/hooks.client.ts` wires it to
 * the browser.
 *
 * The loop guard is a timestamp in sessionStorage (per tab, gone when the tab
 * closes, and reconstructible: losing it only costs one extra reload).
 */

export const RELOAD_GUARD_KEY = 'cropwright.chunkReloadAt';

/** A second chunk failure inside this window is not retried again. */
export const RELOAD_WINDOW_MS = 30_000;

const CHUNK_ERROR_PATTERNS = [
  /failed to fetch dynamically imported module/i, // Chromium
  /error loading dynamically imported module/i, // Firefox
  /importing a module script failed/i, // Safari
  /unable to preload css/i, // Vite's CSS preload helper
  /failed to load module script/i, // wrong MIME type, e.g. index.html served for a missing chunk
  /ChunkLoadError/,
];

export function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;
  if (error && typeof error === 'object' && 'message' in error) {
    return String((error as { message: unknown }).message);
  }
  return '';
}

export function isChunkLoadError(error: unknown): boolean {
  const message = errorMessage(error);
  return CHUNK_ERROR_PATTERNS.some((p) => p.test(message));
}

/** True when no automatic reload has happened within the window. A stamp in
 * the future (clock moved backwards) counts as recent, so it cannot loop. */
export function mayAutoReload(
  lastReloadAt: number | null,
  now: number,
  windowMs: number = RELOAD_WINDOW_MS,
): boolean {
  if (lastReloadAt === null || !Number.isFinite(lastReloadAt)) return true;
  const elapsed = now - lastReloadAt;
  return elapsed >= windowMs;
}

export interface GuardStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

/**
 * Reload once per window. Returns true when a reload was triggered, false when
 * the guard says one already happened recently (the caller then shows the
 * error page). Unreadable storage fails closed: no reload, so it cannot loop.
 */
export function recoverFromChunkError(
  storage: GuardStorage | null,
  now: number,
  reload: () => void,
  windowMs: number = RELOAD_WINDOW_MS,
): boolean {
  if (!storage) return false;
  try {
    const raw = storage.getItem(RELOAD_GUARD_KEY);
    const last = raw === null ? null : Number(raw);
    if (!mayAutoReload(last, now, windowMs)) return false;
    storage.setItem(RELOAD_GUARD_KEY, String(now));
  } catch {
    return false;
  }
  reload();
  return true;
}
