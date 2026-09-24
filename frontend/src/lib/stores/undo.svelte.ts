/**
 * UndoStore — ring buffer of the last 50 human class-write *actions*.
 *
 * One entry per confirmed write, however many crops it touched — a bulk
 * label, a move, or a new-class-proposal resolve over N crops is ONE
 * entry, so one Z reverses the whole action. Calling code records the
 * crop ids a confirmed write touched via `recordWrites(updatedIds)`; Z
 * pops the newest entry and asks the backend to undo it (single-crop or
 * batch route depending on how many ids it holds). The backend owns what
 * "undo" restores and returns the restored item(s) for the page to
 * render.
 */

import { ApiError, undoCropLabel, undoLabelBatch } from '$lib/api';
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
   * Record the crops a human class write just landed on, as ONE undo
   * entry for the whole write. Call with the server's own `updated_ids`
   * (never the request ids minus conflicts computed locally) — the
   * served list is the only authoritative record of which crops the
   * write actually reached. Skipped entirely when the write reached no
   * crop (every id conflicted), so Z never pops a no-op entry.
   */
  recordWrites(updatedIds: string[]): void {
    if (updatedIds.length === 0) return;
    this.push({ crop_ids: [...updatedIds], at: Date.now() });
  }

  /**
   * Z: undo the newest entry on the server and return the restored
   * crop(s) — a single id goes through `POST /crops/{id}/label/undo`, an
   * entry with several goes through the batch
   * `POST /crops/label/undo_batch` — or `[]` when there was nothing to
   * undo or the call failed (both are toasted here). A failed call
   * re-pushes the entry so Z stays retryable; a 409 (nothing left to
   * undo, for either route) does not, since the server has nothing left
   * for it.
   */
  async undoLast(): Promise<Crop[]> {
    const entry = this.pop();
    if (!entry) {
      toastStore.info('Nothing to undo.');
      return [];
    }
    try {
      if (entry.crop_ids.length === 1) {
        const crop = await undoCropLabel(entry.crop_ids[0]!);
        toastStore.success('Reverted.');
        return [crop];
      }
      const res = await undoLabelBatch(entry.crop_ids);
      const parts = [`Reverted ${res.undone}.`];
      if (res.nothing_to_undo.length > 0) {
        parts.push(`${res.nothing_to_undo.length} nothing to undo.`);
      }
      if (res.conflicts.length > 0) {
        parts.push(`${res.conflicts.length} conflict(s).`);
      }
      toastStore.success(parts.join(' '));
      return res.items;
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        toastStore.info('Nothing left to undo.');
      } else {
        toastStore.error(`Undo failed: ${(e as Error).message}`);
        this.push(entry);
      }
      return [];
    }
  }
}

export const undoStore = new UndoStore();
