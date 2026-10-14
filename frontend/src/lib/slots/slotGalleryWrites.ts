/**
 * Region writes of the slot-gallery controller: item-level bulk status,
 * per-box triage, the bbox-editor save patch, and the undo / reprocess
 * merges. Every write renders what the server returned.
 */

import { apiErrorText } from '$lib/api_errors';
import { batchRegionStatus, postBatchBoxState } from '$lib/api_regionBoxes';
import type { RegionBrowseItem } from '$lib/api_regions';
import type { Pager } from '$lib/pager.svelte';
import type { Selection } from '$lib/selection.svelte';
import type { Crop } from '$lib/types';
import { toastStore } from '$stores/toast.svelte';
import { undoStore } from '$stores/undo.svelte';
import type { SlotSpec } from '$lib/annotations/types';

/** The controller state the writes read and set. */
export interface SlotGalleryWriteState {
  readonly slot: SlotSpec;
  readonly pager: Pager<RegionBrowseItem>;
  readonly sel: Selection;
  busy: boolean;
  editCrop: Crop | null;
}

export function createSlotGalleryWrites(s: SlotGalleryWriteState) {
  const { slot, pager, sel } = s;

  /** The distinct item ids behind a set of selected rows (the selection is
   *  per row; an item-level status write is per item). */
  function cropIdsOfRows(rowKeys: string[]): string[] {
    const keySet = new Set(rowKeys);
    const ids = pager.items.filter((p) => keySet.has(p.row_key)).map((p) => p.crop_id);
    return ids.filter((id, i) => ids.indexOf(id) === i);
  }

  async function applyStatus(
    cropIds: string[],
    status: string | undefined,
  ): Promise<void> {
    if (cropIds.length === 0 || s.busy || !status) return;
    s.busy = true;
    sel.clear();
    try {
      // The server validates `status` against its own /regions/statuses.
      // `region_verified` is not sent — the server derives it from
      // `region_status` and ignores the field when present.
      const res = await batchRegionStatus(slot, cropIds, status);
      // Render exactly what the server wrote. `items` covers every crop
      // actually updated; conflicted/invalid ids are left untouched here
      // and reported in the toast below.
      const byId = new Map(res.items.map((p) => [p.crop_id, p]));
      pager.items = pager.items.map((p) => byId.get(p.crop_id) ?? p);
      // M6: Z reverses this bulk status write server-side — the server's
      // own `items` list (not the request's `cropIds`) is what actually
      // got written, same "server's ids, not the request's" rule as
      // every other undoStore.record* call.
      undoStore.recordRegionWrites(res.items.map((p) => p.crop_id));
      const conflictCount = res.conflicts?.length ?? 0;
      const invalid = res.invalid ?? [];
      if (conflictCount > 0 || invalid.length > 0) {
        // a conflict now carries a served `message` (RegionBatchConflict)
        // on a current backend — show it verbatim instead of just a count when
        // present, since it names the real reason ("The item changed
        // since it was loaded."). An older backend has no `message`, so
        // this falls back to the bare count exactly as before.
        const conflictDetail = res.conflicts?.find((c) => c.message)?.message;
        const parts = [
          conflictCount > 0
            ? `${conflictCount} conflicted${conflictDetail ? ` (${conflictDetail})` : ''}`
            : null,
          invalid.length > 0
            ? `${invalid.length} invalid (${invalid.map((i) => i.detail).join('; ')})`
            : null,
        ].filter((s): s is string => s != null);
        toastStore.error(
          `${status.replace('_', ' ')}: ${res.updated} updated, ${parts.join(', ')}`,
        );
      } else {
        toastStore.success(
          `${status.replace('_', ' ')}: ${res.updated} ${slot.label.singular}(s)`,
        );
      }
    } catch (err) {
      toastStore.error(`Bulk update failed: ${apiErrorText(err)}`);
    } finally {
      s.busy = false;
    }
  }

  /**
   * per-box triage over the selected rows (region gallery triage — a
   * cluster is a set of boxes). Unlike `applyStatus` above
   * (whole-item `region_status`, `PATCH region_meta` / `POST
   * batch_status`), this goes through `POST /regions/batch_box_state`,
   * which flips only the targeted box on each item, never its siblings —
   * exactly the spec's rule for triage from a cluster ("never the
   * item-level batch_status, which would flip every sibling box").
   * Targets are built from the row's own served `region_box_id`
   * (`RegionBrowseItem`) — a row without one is skipped
   * rather than silently flipping a whole item.
   */
  async function applyBoxState(rowKeys: string[], state: string): Promise<void> {
    if (rowKeys.length === 0 || s.busy) return;
    const keySet = new Set(rowKeys);
    const targets = pager.items
      .filter((p) => keySet.has(p.row_key) && p.region_box_id != null)
      .map((p) => ({ cropId: p.crop_id, boxId: p.region_box_id! }));
    const skipped = rowKeys.length - targets.length;
    if (targets.length === 0) {
      toastStore.error(
        `Cannot triage by box: this row has no served box id (pre-W8 backend?).`,
      );
      return;
    }
    s.busy = true;
    sel.clear();
    try {
      const res = await postBatchBoxState(targets, state);
      // A box-state write returns full items (Crop), not gallery rows —
      // the row grid re-fetches its own page rather than patching rows
      // in place from a shape the gallery doesn't render (RegionBrowseItem
      // vs Crop): the triaged boxes typically leave the current bucket
      // anyway (their state changed), so a refetch is the correct result,
      // not a shortcut.
      undoStore.recordRegionWrites(res.items.map((c) => c.id));
      const invalid = res.invalid ?? [];
      const conflicts = res.conflicts ?? [];
      if (invalid.length > 0 || conflicts.length > 0) {
        const conflictDetail = conflicts.find((c) => c.message)?.message;
        const parts = [
          conflicts.length > 0
            ? `${conflicts.length} conflicted${conflictDetail ? ` (${conflictDetail})` : ''}`
            : null,
          invalid.length > 0
            ? `${invalid.length} invalid (${invalid.map((i) => i.message).join('; ')})`
            : null,
          skipped > 0 ? `${skipped} skipped (no box id)` : null,
        ].filter((s): s is string => s != null);
        toastStore.error(
          `${state.replace('_', ' ')}: ${res.updated} updated, ${parts.join(', ')}`,
        );
      } else {
        toastStore.success(
          `${state.replace('_', ' ')}: ${res.updated} box(es)${skipped > 0 ? `, ${skipped} skipped (no box id)` : ''}`,
        );
      }
      await pager.loadPage(pager.firstPage);
    } catch (err) {
      toastStore.error(`Bulk box triage failed: ${apiErrorText(err)}`);
    } finally {
      s.busy = false;
    }
  }

  /**
   * `onsave` for `SlotBboxEditor` — the editor has ALREADY performed the
   * write (`PUT /crops/{id}/regions`) by the time this fires, and passes back the
   * server's own returned item. This function only patches the matching
   * card from that item; it must NOT re-PUT the box (a pre-C8 bug — this
   * used to write the box a second time here, redundantly re-sending a
   * box the editor had just saved), and must NOT re-derive
   * confirmed-vs-rejected status client-side — it renders what the
   * server wrote.
   */
  function saveBox(item: Crop): void {
    if (!s.editCrop) return;
    const cropId = item.id;
    toastStore.success(`${slot.label.title} saved`);
    s.editCrop = null;
    const slotData = item.slots?.[slot.key];
    // Patch just this card in place rather than reloading page 1 (which
    // would wipe the list and reset scroll). The returned item's slot data
    // carries the new boxes and revision, which is also what re-crops the
    // card's thumbnail.
    pager.items = pager.items.map((p) =>
      p.crop_id === cropId
        ? {
            ...p,
            region_status: slotData?.lifecycle?.status ?? p.region_status,
            region_verified: slotData?.lifecycle?.verified ?? p.region_verified,
            slots: { ...p.slots, ...item.slots },
          }
        : p,
    );
    // M6: the editor's write (already completed by the time this fires —
    // see the doc comment above) is undoable via Z too.
    undoStore.recordRegionWrites([cropId]);
  }

  /**
   * M6: Z on the region gallery reverses the most recent region write
   * (single-item bbox edit or bulk status change), the same way Z
   * reverses a label write on the card-grid `/clusters` view — see
   * `undo.svelte.ts`'s header comment for why this shares the ONE
   * `undoStore` stack (kind: `'region'`) with every other undo entry
   * rather than its own gallery-local stack.
   */
  function mergeUndoneItems(crops: Crop[]): void {
    if (crops.length === 0) return;
    const byId = new Map(crops.map((c) => [c.id, c]));
    pager.items = pager.items.map((p) => {
      const restored = byId.get(p.crop_id);
      if (!restored) return p;
      const slotData = restored.slots?.[slot.key];
      return {
        ...p,
        region_status: slotData?.lifecycle?.status ?? p.region_status,
        region_verified: slotData?.lifecycle?.verified ?? p.region_verified,
        slots: { ...p.slots, ...restored.slots },
      };
    });
  }

  /**
   * `onreprocessed` for a card's image Reprocess: the served items are
   * merged into the cards already loaded (later pages included), then the
   * first page is re-fetched, since re-detection can add, move or drop the
   * regions an image contributes.
   */
  async function adoptReprocessed(items: Crop[]): Promise<void> {
    mergeUndoneItems(items);
    await pager.loadPage(pager.firstPage);
  }

  async function undoLastAction(): Promise<void> {
    const crops = await undoStore.undoLast();
    mergeUndoneItems(crops);
  }

  return {
    cropIdsOfRows,
    applyStatus,
    applyBoxState,
    saveBox,
    adoptReprocessed,
    undoLastAction,
  };
}
