/**
 * Pure queue-mutation helpers extracted from `review/+page.svelte`'s
 * slot-tab undo/back-navigation logic (P0.1,
 * docs/genericization-plan-2026-09-13.md §5.1/§5a).
 *
 * These were inline closures over the page's undo stack/`queue.items` state;
 * extracted as pure functions so they're independently testable and so
 * Phase 2 can parameterize them by slot without touching the review
 * page's control flow. Behavior is unchanged — every function here is a
 * direct lift of the logic that used to live at
 * `review/+page.svelte:737-780`.
 */

/**
 * Bounded, FIFO-evicting push. `$state.raw` in the component (NOT
 * `$state`) is what makes `removeUndo`'s identity filter below work —
 * deep reactivity would proxy every pushed entry, so `e !== entry` could
 * never match the object the caller holds. That's a property of how the
 * component stores the array, not of this function, so it's noted here
 * for whoever wires this back in.
 */
export function pushUndo<T>(stack: readonly T[], entry: T, max: number): T[] {
  return [...stack, entry].slice(-max);
}

/** Drop a specific entry by object identity — used when its paired API
 *  call failed and the optimistic undo entry must not be replayable. */
export function removeUndo<T>(stack: readonly T[], entry: T): T[] {
  return stack.filter((e) => e !== entry);
}

/** Pop the most recent entry. Returns `entry: undefined` on an empty
 *  stack rather than throwing — callers decide what "nothing to undo"
 *  means (e.g. a toast). */
export function popUndo<T>(stack: readonly T[]): { entry: T | undefined; rest: T[] } {
  if (stack.length === 0) return { entry: undefined, rest: [] };
  return { entry: stack[stack.length - 1], rest: stack.slice(0, -1) };
}

/**
 * Re-insert `item` into `items` at `insertAt`, clamped to the current
 * length so an index captured before other mutations landed (e.g. more
 * items loaded, or the queue shrank) never throws or silently drops the
 * item off the end.
 */
export function reinsertAt<T>(items: readonly T[], insertAt: number, item: T): T[] {
  const idx = Math.min(insertAt, items.length);
  const next = [...items];
  next.splice(idx, 0, item);
  return next;
}
