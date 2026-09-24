/**
 * Per-key in-flight-request registry, extracted from `review/+page.svelte`'s
 * slot-metadata abort map (P0.3, docs/genericization-plan-2026-09-13.md
 * §5.1/§5a).
 *
 * Keyed by crop id (not cursor position) so that if the operator advances
 * to a different crop mid-save, the abort/settle bookkeeping for the
 * earlier save can never be misattributed to whatever crop the cursor
 * now points at. A second `start(id)` for the same id aborts the first;
 * `start` for a different id never touches the first id's controller.
 */
export class AbortRegistry {
  private readonly controllers = new Map<string, AbortController>();

  /** Aborts any in-flight request for `id`, then registers and returns a
   *  fresh controller for it. */
  start(id: string): AbortController {
    this.controllers.get(id)?.abort();
    const ac = new AbortController();
    this.controllers.set(id, ac);
    return ac;
  }

  /** True iff `ac` is still the current (non-superseded) controller for
   *  `id` — check this after an await to decide whether a response is
   *  stale and should be discarded rather than applied. */
  isCurrent(id: string, ac: AbortController): boolean {
    return this.controllers.get(id) === ac;
  }

  /** Remove `ac` from the registry iff it's still the current entry for
   *  `id` (a superseding `start()` may have already replaced it, in
   *  which case this must be a no-op or it would erase the newer
   *  controller). Call from a `finally` block after the request settles. */
  finish(id: string, ac: AbortController): void {
    if (this.controllers.get(id) === ac) this.controllers.delete(id);
  }
}
