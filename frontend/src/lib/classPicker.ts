/**
 * Logic for the `/review` (and eventually `/clusters/[id]`) type-to-search
 * class picker — audit remediation plan Phase 7, P1-4/P1-5.
 *
 * `quickAssignClasses` caps quick-assign at the 10
 * most-validated classes; 73 of 84 classes require a round-trip to
 * `/classes` to become assignable, which is exactly where the labeling
 * debt concentrates (44 classes at zero validated, including brand-new
 * ones like `scion` that have zero samples and therefore can never climb
 * into the top 10 on their own). This module is the pure filter/ranking
 * logic behind a `/`-opened combobox that searches *all* non-deprecated
 * classes, extracted out of `+page.svelte` per this repo's "no
 * @testing-library/svelte — test the logic, not the rendered component"
 * convention (see `pager.svelte.ts`, `selection.svelte.ts`,
 * `strategyBar.svelte.ts`).
 *
 * Kept deliberately free of any Svelte/store import so it's trivially
 * unit-testable with plain arrays.
 */

import { isAssignableClass, isItemClassTarget } from '$lib/classVisibility';
import type { RegistryClass } from '$lib/types';

/** Rank buckets, lowest = best match. Exact match beats prefix beats
 *  substring beats subsequence ("fuzzy"); anything else doesn't match. */
function matchRank(name: string, query: string): number | null {
  if (name === query) return 0;
  if (name.startsWith(query)) return 1;
  if (name.includes(query)) return 2;
  if (isSubsequence(name, query)) return 3;
  return null;
}

/** True if every character of `query`, in order, appears somewhere in
 *  `name` (not necessarily contiguous) — the classic "fzf-lite" fallback
 *  for typos / abbreviations once prefix/substring both miss. */
function isSubsequence(name: string, query: string): boolean {
  let i = 0;
  for (let j = 0; j < name.length && i < query.length; j++) {
    if (name[j] === query[i]) i++;
  }
  return i === query.length;
}

/**
 * Search every non-deprecated class by name. Empty query returns the full
 * non-deprecated set ordered by validated_count desc (same ordering
 * `quickAssignClasses` uses, just not truncated to 10) so opening the picker
 * with no query still shows something useful before the operator types.
 *
 * `limit` truncates the *rendered* list (the combobox doesn't want an
 * 84-row dropdown by default); omit it to get every match, which is what
 * proves "all 84 classes, not top 10" in tests.
 */
export function searchClasses(
  classes: RegistryClass[],
  query: string,
  limit?: number,
  prefer: ReadonlyArray<number | null | undefined> = [],
): RegistryClass[] {
  const pool = classes.filter(isAssignableClass);
  const q = query.trim().toLowerCase();

  let ranked: RegistryClass[];
  if (!q) {
    ranked = preferFirst(
      [...pool].sort(
        (a, b) =>
          (b.validated_count ?? 0) - (a.validated_count ?? 0) ||
          a.name.localeCompare(b.name),
      ),
      prefer,
    );
  } else {
    ranked = pool
      .map((cls) => ({ cls, rank: matchRank(cls.name.toLowerCase(), q) }))
      .filter((s): s is { cls: RegistryClass; rank: number } => s.rank !== null)
      .sort(
        (a, b) =>
          a.rank - b.rank ||
          (b.cls.validated_count ?? 0) - (a.cls.validated_count ?? 0) ||
          a.cls.name.localeCompare(b.cls.name),
      )
      .map((s) => s.cls);
  }

  return limit != null ? ranked.slice(0, limit) : ranked;
}

/** Moves every class whose id is in `prefer` to the front, in `prefer`'s
 *  order (first occurrence wins), keeping the rest in their given order. */
function preferFirst(
  ranked: RegistryClass[],
  prefer: ReadonlyArray<number | null | undefined>,
): RegistryClass[] {
  const ids = [...new Set(prefer.filter((id): id is number => id != null))];
  if (ids.length === 0) return ranked;
  const front = ids
    .map((id) => ranked.find((c) => c.id === id))
    .filter((c): c is RegistryClass => c != null);
  return [...front, ...ranked.filter((c) => !ids.includes(c.id))];
}

/** The served class ids an item already points at — what it is proposed
 *  as, what it currently is, what the VLM and the model suggest — in
 *  that order. Used to rank the picker/quick-assign row for THIS item
 *  instead of by global validated count (visual audit 2026-09-24, R1). */
export function itemHintClassIds(
  item:
    | {
        proposed_class_id?: number | null;
        class_id?: number | null;
        vlm_suggested_class_id?: number | null;
        probe_pred_class_id?: number | null;
      }
    | null
    | undefined,
): number[] {
  if (!item) return [];
  return [
    item.proposed_class_id,
    item.class_id,
    item.vlm_suggested_class_id,
    item.probe_pred_class_id,
  ].filter((id): id is number => id != null);
}

/** Every class an item may be labeled as — see `isItemClassTarget`. */
export function itemClassTargets(classes: RegistryClass[]): RegistryClass[] {
  return classes.filter(isItemClassTarget);
}

/**
 * The `/review` quick-assign row: item-class targets only (never a
 * slot-bound region class — `isItemClassTarget`), with the item's own
 * hinted classes first and the most-validated classes filling the rest.
 */
export function quickAssignClasses(
  classes: RegistryClass[],
  prefer: ReadonlyArray<number | null | undefined>,
  n = 10,
): RegistryClass[] {
  return searchClasses(itemClassTargets(classes), '', undefined, prefer).slice(0, n);
}

/**
 * Which class id `Enter`/Confirm would assign for a review item, or null
 * when there's nothing to confirm. Mirrors the audit's "corrected" reading
 * (P1-5): the field that matters is `proposed_class_id`, falling back to
 * the crop's *current* `class_id` — never `proposed_class_name`, which is
 * display-only.
 *
 * Extracted so `+page.svelte`'s Confirm button/Enter handler and the
 * `canConfirm` guard share one source of truth instead of each re-deriving
 * "is there anything to confirm" separately.
 */
export function resolveConfirmClassId(
  item: { proposed_class_id: number | null } | null | undefined,
): number | null {
  // The backend decides what is confirmable: proposed_class_id is the
  // item's class unless there is nothing to confirm (e.g. a VLM
  // new-class proposal), where it is null and Enter opens the picker.
  return item?.proposed_class_id ?? null;
}
