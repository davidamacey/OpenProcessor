import { describe, it, expect } from 'vitest';
import { pushUndo, removeUndo, popUndo, reinsertAt } from './slotQueueOps';

describe('pushUndo', () => {
  it('appends and evicts FIFO once past the bound', () => {
    let stack: number[] = [];
    for (let i = 0; i < 25; i++) stack = pushUndo(stack, i, 20);
    expect(stack).toHaveLength(20);
    expect(stack[0]).toBe(5); // 0-4 evicted
    expect(stack[stack.length - 1]).toBe(24);
  });

  it('does not mutate the input array', () => {
    const original = [1, 2, 3];
    const next = pushUndo(original, 4, 20);
    expect(original).toEqual([1, 2, 3]);
    expect(next).toEqual([1, 2, 3, 4]);
  });
});

describe('removeUndo', () => {
  it('removes by object identity, not by structural equality', () => {
    const a = { id: 'x' };
    const b = { id: 'x' }; // structurally equal, different object
    const stack = [a, b];
    expect(removeUndo(stack, a)).toEqual([b]);
    expect(removeUndo(stack, a)).not.toContain(a);
  });

  it('is a no-op when the entry is not present', () => {
    const a = { id: 'x' };
    const stack = [{ id: 'y' }];
    expect(removeUndo(stack, a)).toEqual(stack);
  });
});

describe('popUndo', () => {
  it('returns the last entry and the remaining stack', () => {
    const { entry, rest } = popUndo([1, 2, 3]);
    expect(entry).toBe(3);
    expect(rest).toEqual([1, 2]);
  });

  it('returns undefined entry and an empty rest on an empty stack', () => {
    const { entry, rest } = popUndo([]);
    expect(entry).toBeUndefined();
    expect(rest).toEqual([]);
  });
});

describe('reinsertAt', () => {
  it('inserts at the given index when in range', () => {
    expect(reinsertAt(['a', 'c'], 1, 'b')).toEqual(['a', 'b', 'c']);
  });

  it('clamps an out-of-range index to the end of the array (queue shrank since insertAt was captured)', () => {
    expect(reinsertAt(['a', 'b'], 99, 'z')).toEqual(['a', 'b', 'z']);
  });

  it('does not mutate the input array', () => {
    const original = ['a', 'b'];
    reinsertAt(original, 1, 'x');
    expect(original).toEqual(['a', 'b']);
  });
});
