/**
 * Unit tests for the UndoStore ring buffer.
 *
 * The store is a module singleton, so every test clears it first.
 */

import { beforeEach, describe, expect, it } from 'vitest';
import { undoStore } from './undo.svelte';
import type { UndoEntry } from '$lib/types';

function entry(id: string): UndoEntry {
  return {
    crop_id: id,
    prior_class_id: 1,
    prior_label_source: 'model_suggestion',
    prior_validated: false,
    at: 0,
  };
}

describe('undoStore', () => {
  beforeEach(() => {
    undoStore.clear();
  });

  it('pops in LIFO order', () => {
    undoStore.push(entry('a'));
    undoStore.push(entry('b'));
    undoStore.push(entry('c'));
    expect(undoStore.pop()?.crop_id).toBe('c');
    expect(undoStore.pop()?.crop_id).toBe('b');
    expect(undoStore.pop()?.crop_id).toBe('a');
  });

  it('caps at 50 entries, dropping the oldest', () => {
    for (let i = 0; i < 51; i++) undoStore.push(entry(`e${i}`));
    expect(undoStore.stack).toHaveLength(50);
    expect(undoStore.stack[0]!.crop_id).toBe('e1');
    expect(undoStore.stack[49]!.crop_id).toBe('e50');
  });

  it('returns undefined when popping an empty stack', () => {
    expect(undoStore.pop()).toBeUndefined();
  });

  it('remove() drops exactly the given entries by identity', () => {
    const a = entry('a');
    const b = entry('b');
    const c = entry('c');
    undoStore.push(a);
    undoStore.push(b);
    undoStore.push(c);
    undoStore.remove([a, c]);
    expect(undoStore.stack.map((e) => e.crop_id)).toEqual(['b']);
  });

  it('remove() ignores entries that are no longer on the stack', () => {
    const a = entry('a');
    undoStore.push(a);
    undoStore.remove([entry('a')]); // structurally equal but a different object
    expect(undoStore.stack).toHaveLength(1);
  });

  it('snapshotOf() captures the prior label state of a crop', () => {
    const snap = undoStore.snapshotOf({
      id: 'crop-1',
      class_id: 7,
      label_source: 'gemma_suggestion',
      label_validated: true,
    });
    expect(snap.crop_id).toBe('crop-1');
    expect(snap.prior_class_id).toBe(7);
    expect(snap.prior_label_source).toBe('gemma_suggestion');
    expect(snap.prior_validated).toBe(true);
    expect(typeof snap.at).toBe('number');
  });
});
