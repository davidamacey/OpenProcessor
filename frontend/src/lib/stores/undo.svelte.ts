/**
 * UndoStore — ring buffer of the last 50 label actions.
 *
 * Calling code is expected to push the prior state of a crop *before* the
 * mutation goes out, so undo can restore by re-issuing PUT or DELETE.
 */

import type { UndoEntry } from '$lib/types';

const MAX = 50;

class UndoStore {
  stack = $state<UndoEntry[]>([]);

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

  clear(): void {
    this.stack = [];
  }
}

export const undoStore = new UndoStore();
