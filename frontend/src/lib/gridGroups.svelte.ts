/**
 * Grid-group state for the /clusters/[id] crop grid.
 *
 * The grid renders one `svelte-dnd-action` dndzone per sub-cluster group, so
 * the rendered list can't be `cropPager.items` itself: the dnd library owns
 * the zone's item order for the duration of a drag and hands it back through
 * `consider`/`finalize` events. The page used to hold that rendered list in a
 * plain `$state` snapshot rebuilt by an `$effect`, and let the dnd event
 * handlers assign into the very same `$state`.
 *
 * That is the "moved crops flicker back and stay" bug (2026-09-12 live report,
 * confirmed in `op_vehicle_crops.class_id_history`: two crops were batch
 * labelled 15 -> 72 at 04:57:22 and then labelled 72 -> 72 *again*, one at a
 * time, a minute later — the operator re-dragging crops the grid was still
 * showing even though the server had already moved them). The snapshot had no
 * invariant tying it back to the pager:
 *
 *  - `consider`/`finalize` carry the *library's* copy of the zone list, which
 *    is a pre-drop snapshot. Instrumented runs show `finalize` arriving with
 *    N+1 items while the pager already holds N — i.e. the grid is routinely
 *    repainted from a list that still contains a crop the drop handler just
 *    labelled away.
 *  - Nothing re-derived the snapshot afterwards. The rebuild `$effect` only
 *    re-runs when the pager list (or the grouping mode) changes, so a stale
 *    dnd-sourced write that lands *after* the optimistic removal is the last
 *    word until a hard page reload.
 *
 * The fix makes the invariant structural instead of incidental:
 *
 *  1. The authoritative grouping is a real `$derived` of the pager list. It
 *     can never be stale with respect to the source of truth.
 *  2. A dnd event may install a *transient* override (so the in-flight drag
 *     shuffle is visible), but that override is reconciled against the live
 *     id set on the way in — an item the pager no longer holds can never be
 *     painted back into the grid.
 *  3. `reset()` drops the override, so the derived truth takes over again.
 *     Every drag end and every optimistic mutation resets, which means a
 *     divergence can survive at most until the end of the current gesture
 *     rather than until a reload.
 */

export interface GridGroup<T> {
  key: string;
  label: string;
  items: T[];
}

export interface GridGroupsOptions<T> {
  /** The list the grid should show — `filteredCrops` on the cluster page. */
  source: () => T[];
  /** Whether to partition into contiguous sub-cluster groups. */
  grouped: () => boolean;
  /** Stable identity. */
  keyOf: (item: T) => string;
  /** Sub-cluster bucket, or null for "unrefined". */
  subidOf: (item: T) => string | null;
  /**
   * Ids the page still considers present. A dnd event list is filtered
   * against this before it is allowed to drive the grid, which is what stops
   * a post-drop `consider`/`finalize` from resurrecting a labelled-away crop.
   * Defaults to the ids in `source()`.
   */
  liveIds?: () => Set<string>;
}

export interface GridGroupsState<T> {
  /** What the template renders. */
  readonly groups: GridGroup<T>[];
  /** True while a dnd-sourced override is installed. */
  readonly overridden: boolean;
  /** Adopt the dnd library's list for one zone (reconciled first). */
  setZoneItems(key: string, items: T[]): void;
  /** Discard the override — the derived grouping becomes authoritative. */
  reset(): void;
}

/**
 * Partition `source` into render groups.
 *
 * Ungrouped: one `__all__` group holding a copy of the whole list (never hand
 * the caller's array to a zone — the dnd action treats it as its own).
 * Grouped: sort by sub-cluster id (unrefined last) and walk contiguous runs.
 */
export function buildGroups<T>(
  source: T[],
  grouped: boolean,
  subidOf: (item: T) => string | null,
): GridGroup<T>[] {
  if (!grouped) {
    return [{ key: '__all__', label: '', items: [...source] }];
  }
  const sorted = [...source].sort((a, b) => {
    const sa = subidOf(a) ?? '￿';
    const sb = subidOf(b) ?? '￿';
    return sa < sb ? -1 : sa > sb ? 1 : 0;
  });
  const groups: GridGroup<T>[] = [];
  let lastSub: string | null = null;
  for (const c of sorted) {
    const sub = subidOf(c) ?? '__none__';
    const last = groups[groups.length - 1];
    if (!last || lastSub !== sub) {
      // The key carries the run index, not just the subid: this is a
      // contiguity walk, so a subid that reappears non-contiguously (a
      // transient render over a not-yet-sorted list) would emit the same key
      // twice and crash the keyed {#each} with each_key_duplicate.
      groups.push({
        key: `${sub}#${groups.length}`,
        label: sub === '__none__' ? 'unrefined' : `sub-cluster ${sub}`,
        items: [c],
      });
      lastSub = sub;
    } else {
      last.items.push(c);
    }
  }
  return groups;
}

export function createGridGroups<T>(opts: GridGroupsOptions<T>): GridGroupsState<T> {
  const derivedGroups = $derived.by(() =>
    buildGroups(opts.source(), opts.grouped(), opts.subidOf),
  );

  // Transient dnd-owned view of the grid. Non-null only between a `consider`
  // and the matching `reset()`.
  let override = $state<GridGroup<T>[] | null>(null);

  const groups = $derived(override ?? derivedGroups);

  function live(): Set<string> {
    return opts.liveIds ? opts.liveIds() : new Set(opts.source().map(opts.keyOf));
  }

  return {
    get groups() {
      return groups;
    },
    get overridden() {
      return override !== null;
    },
    setZoneItems(key: string, items: T[]): void {
      const base = groups;
      const idx = base.findIndex((g) => g.key === key);
      // The zone's group is gone (the derived grouping was rebuilt under an
      // in-flight drag). Nothing to adopt — the derived truth already covers
      // whatever the event was trying to say.
      if (idx === -1) return;
      const ids = live();
      const seen = new Set<string>();
      const reconciled = items.filter((it) => {
        const id = opts.keyOf(it);
        // Drop ids the page no longer holds (already labelled/moved/discarded
        // away) and any duplicate — a repeated key crashes the keyed {#each}.
        if (!ids.has(id) || seen.has(id)) return false;
        seen.add(id);
        return true;
      });
      override = base.map((g, i) => (i === idx ? { ...g, items: reconciled } : g));
    },
    reset(): void {
      override = null;
    },
  };
}
