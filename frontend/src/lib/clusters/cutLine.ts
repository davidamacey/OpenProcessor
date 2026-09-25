/**
 * DQ-M3 frontend half (docs/design/data-quality-pass-2026-09-24.md):
 * `cluster_is_core` is null for most (about 85%) class-cluster members, and
 * `/crops?cluster_id=N` defaults to `sort=updated_at:desc` — NOT
 * core-first — while `clusters/[id]/+page.svelte`'s old `cutLineIndex`
 * assumed the API had already sorted crops core-first. Live: a class
 * cluster's member #1 is usually not core, so the old logic (stop at the
 * first non-core-true item) degenerately produced index 0 there — but for
 * candidate clusters, EVERY member has `cluster_is_core` set, just not in
 * core-first order, so the old logic drew a line at some arbitrary
 * non-core-to-core boundary that meant nothing (#10000's line landed at
 * index 181 with core crops resuming right after it at 182).
 *
 * There is no server-side "core first" order to request instead (checked
 * against contracts/openprocessor/openapi/curation.json — no such sort
 * id exists on `/crops` or `/clusters/{id}`'s crop listing) — so per the
 * fix list ("hide the line rather than guessing"), this computes whether
 * the currently-loaded order actually IS core-first-consistent before
 * trusting a boundary index, instead of assuming it.
 */

export interface CutLineCropLike {
  cluster_is_core?: boolean | null;
}

export interface CutLineResult {
  /** Index of the first non-core crop, or crops.length if every crop is core. */
  index: number;
  /** Whether drawing a line at `index` is trustworthy. */
  visible: boolean;
}

/** Below this null-share, `cluster_is_core` is too sparse on this view to
 *  mean anything — matches the ~85% null rate measured
 *  on class clusters, comfortably over half. */
const MOSTLY_NULL_THRESHOLD = 0.5;

export function computeCutLine(crops: readonly CutLineCropLike[]): CutLineResult {
  if (crops.length === 0) return { index: 0, visible: false };

  const nullCount = crops.filter((c) => c.cluster_is_core == null).length;
  if (nullCount / crops.length > MOSTLY_NULL_THRESHOLD) {
    return { index: 0, visible: false };
  }

  // Find the first non-core crop, then verify no LATER crop is core:true —
  // that would mean the loaded order isn't actually core-first (the
  // candidate-cluster case above), and any boundary drawn on top of it is
  // arbitrary rather than a real "core ends here" line.
  let cutIndex = crops.length;
  let orderIsCoreFirst = true;
  for (let i = 0; i < crops.length; i++) {
    const isCore = crops[i].cluster_is_core === true;
    if (cutIndex === crops.length) {
      if (!isCore) cutIndex = i;
    } else if (isCore) {
      orderIsCoreFirst = false;
      break;
    }
  }

  if (!orderIsCoreFirst) return { index: 0, visible: false };
  // Nothing to draw: every crop is core (no boundary) or every crop is
  // non-core (boundary at 0 — the degenerate case the old logic already
  // produced for most class clusters, made explicit here).
  if (cutIndex <= 0 || cutIndex >= crops.length)
    return { index: cutIndex, visible: false };
  return { index: cutIndex, visible: true };
}
