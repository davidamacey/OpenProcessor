/**
 * DQ-M4 (docs/design/data-quality-pass-2026-09-24.md): `/clusters`'
 * `representatives_offset`/`representatives_limit` window into the card
 * list only ever windows the backend's own size-desc (`_count desc`)
 * order — there is no batch-by-id representatives param (checked against
 * the vendored contract, contracts/openapi/curation.json:
 * `/curation/clusters` and `/curation/clusters/representatives` both only
 * take a numeric `offset`/`limit` into that fixed server order, plus a
 * single `cluster_id` filter — no `cluster_ids` list). The cluster grid
 * sorts cards client-side (purity/size/dominant_class — the endpoint
 * itself isn't sortable, see sortClusters() in clusters/+page.svelte), so
 * at any sort other than the server's own, the windowed offset filled
 * representatives for a *different* set of cards than the ones actually
 * shown first — 4 of the first 8 cards rendered blank until a scroll
 * dragged the window far enough to cover them.
 *
 * Since there's no batch param, the fix fetches representatives one
 * `cluster_id` at a time (`getClusters({cluster_id})`, small/cheap per
 * the perf table — every non-diverse /clusters call in the audit was
 * under 60ms) for exactly the cluster ids visible in the current DISPLAY
 * order window. `idsNeedingRepresentatives` is the pure selection logic:
 * given the already-sorted display list and a window, return only the
 * ids that (a) aren't a synthetic slot inventory card (no real
 * `cluster_id` to query) and (b) don't already carry representatives —
 * so re-sorting after the first screenful is filled never re-fetches
 * cards it already has thumbnails for.
 */

export interface DisplayCardLike {
  id: number;
  isSlotCard?: boolean;
  representative_crop_ids: string[];
}

export function idsNeedingRepresentatives(
  displayOrder: readonly DisplayCardLike[],
  windowStart: number,
  windowSize: number,
): number[] {
  return displayOrder
    .slice(windowStart, windowStart + windowSize)
    .filter((c) => !c.isSlotCard && c.representative_crop_ids.length === 0)
    .map((c) => c.id);
}
