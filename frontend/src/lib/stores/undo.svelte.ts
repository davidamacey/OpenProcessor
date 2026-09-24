/**
 * UndoStore — ring buffer of the last 50 human class writes.
 *
 * Calling code records each crop a confirmed label write touched. Z pops
 * the newest and asks the backend to undo that crop's most recent human
 * class write; the backend owns what "undo" restores and returns the
 * restored item for the page to render.
 */

import { ApiError, undoCropLabel } from '$lib/api';
import { toastStore } from '$stores/toast.svelte';
import type { Crop, UndoEntry } from '$lib/types';

const MAX = 50;

class UndoStore {
  // $state.raw, not $state: deep reactivity would wrap every pushed entry
  // in a Proxy, so `remove()` could never match the raw object the caller
  // still holds. Every mutation below reassigns the array, so raw state is
  // just as reactive for readers.
  stack = $state.raw<UndoEntry[]>([]);

  push(entry: UndoEntry): void {
    const next = [...this.stack, entry];
    if (next.length > MAX) next.shift();
    this.stack = next;
  }

  pop(): UndoEntry | undefined {
    if (this.stack.length === 0) return undefined;
    const next = [...this.stack];
    const entry = next.pop();
    this.stack = next;
    return entry;
  }

  /**
   * Drop specific entries by identity.
   *
   * Revert paths must use this rather than a bare `pop()`: the global
   * stack can change during an in-flight request (the operator can press
   * Z mid-flight and pop *your* entry), so a blind pop would remove an
   * unrelated action's history instead.
   */
  remove(entries: UndoEntry[]): void {
    if (entries.length === 0) return;
    const s = new Set(entries);
    this.stack = this.stack.filter((e) => !s.has(e));
  }

  clear(): void {
    this.stack = [];
  }

  /**
   * Record the crops a human class write just landed on. Call with the
   * server's own `updated_ids` (never the request ids minus conflicts
   * computed locally) — the served list is the only authoritative record
   * of which crops the write actually reached.
   */
  recordWrites(updatedIds: string[]): void {
    const at = Date.now();
    for (const id of updatedIds) this.push({ crop_id: id, at });
  }

  /**
   * Z: undo the newest entry on the server and return the restored crop,
   * or null when there was nothing to undo or the call failed (both are
   * toasted here). A failed call re-pushes the entry so Z stays
   * retryable; a 409 does not, since the server has nothing left for it.
   */
  async undoLast(): Promise<Crop | null> {
    const entry = this.pop();
    if (!entry) {
      toastStore.info('Nothing to undo.');
      return null;
    }
    try {
      const crop = await undoCropLabel(entry.crop_id);
      toastStore.success('Reverted.');
      return crop;
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        toastStore.info('Nothing left to undo for that crop.');
      } else {
        toastStore.error(`Undo failed: ${(e as Error).message}`);
        this.push(entry);
      }
      return null;
    }
  }
}

export const undoStore = new UndoStore();
