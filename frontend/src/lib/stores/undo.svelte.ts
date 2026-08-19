/**
 * UndoStore — ring buffer of the last 50 label actions.
 *
 * Calling code is expected to push the prior state of a crop *before* the
 * mutation goes out, so undo can restore by re-issuing PUT or DELETE.
 */

import type { LabelSource, UndoEntry } from '$lib/types';

const MAX = 50;

/** The subset of a crop/review item an UndoEntry is built from. */
export interface UndoSnapshotSource {
  id: string;
  class_id: number | null;
  label_source: LabelSource;
  label_validated: boolean;
}

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

  /** Build the pre-mutation snapshot for a crop or review item. */
  snapshotOf(src: UndoSnapshotSource): UndoEntry {
    return {
      crop_id: src.id,
      prior_class_id: src.class_id,
      prior_label_source: src.label_source,
      prior_validated: src.label_validated,
      at: Date.now(),
    };
  }
}

export const undoStore = new UndoStore();
