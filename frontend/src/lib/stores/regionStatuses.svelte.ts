/**
 * RegionStatusesStore — the deployment's region-status vocabulary
 * (`GET {API_PREFIX}/regions/statuses`), loaded once. Drives the slot
 * panel's status dropdown, clear/rejection-reason behavior and
 * confirm/reject/false-positive actions (`$lib/review/slotPanel.ts`)
 * instead of a hand-copied literal per slot profile.
 *
 * On failure (or before `init()` resolves) the list is empty and every
 * `slotPanel.ts` helper falls back to the active slot's own
 * `capabilities.lifecycle.states` — the pre-existing, hand-maintained
 * vocabulary — so a missing endpoint never breaks the review panel.
 */

import { getRegionStatuses, type BoxStateEntry, type RegionStatusEntry } from '$lib/api';

class RegionStatusesStore {
  list = $state<RegionStatusEntry[]>([]);
  confirmStatus = $state<string | null>(null);
  rejectStatus = $state<string | null>(null);
  falsePositiveStatus = $state<string | null>(null);
  /** W8.7: served per-box state vocabulary — distinct from `list` above
   *  (item-level `region_status`). Empty on a pre-W8 backend; every
   *  caller falls back to a hardcoded palette (see `multiBoxRingColor`/
   *  `multiBoxStateLabel` in `/review`, `SourceImageOverlay.svelte`,
   *  `CropMetaPanel.svelte`) when a value isn't found here. */
  boxStates = $state<BoxStateEntry[]>([]);
  loaded = $state<boolean>(false);
  #inflight: Promise<void> | null = null;

  /** The served label for a stored status value, or `null` when the
   *  vocabulary isn't loaded or doesn't know the value (callers fall back
   *  to the slot profile's own label). */
  labelFor(value: string | null | undefined): string | null {
    if (!value) return null;
    return this.list.find((s) => s.value === value)?.label ?? null;
  }

  /** The served `box_states` entry for a box `state` value, or `null`
   *  when unloaded/unknown. */
  boxStateInfo(value: string | null | undefined): BoxStateEntry | null {
    if (!value) return null;
    return this.boxStates.find((s) => s.value === value) ?? null;
  }

  /** The box-state value (not the item-level `region_status`) whose
   *  served `role` is `role` — e.g. `boxStateByRole('accepted')` for the
   *  cluster-triage "Verify" action. `null` when `box_states` hasn't
   *  loaded (pre-W8 backend); callers fall back to the literal role
   *  string, which is the fixed W8.7 wire vocabulary, not a guess. */
  boxStateByRole(role: string): string | null {
    return this.boxStates.find((s) => s.role === role)?.value ?? null;
  }

  async init(): Promise<void> {
    if (this.loaded) return;
    if (this.#inflight) return this.#inflight;
    this.#inflight = (async () => {
      try {
        const res = await getRegionStatuses();
        this.list = res.statuses ?? [];
        this.confirmStatus = res.confirm_status ?? null;
        this.rejectStatus = res.reject_status ?? null;
        this.falsePositiveStatus = res.false_positive_status ?? null;
        this.boxStates = res.box_states ?? [];
      } catch {
        this.list = [];
        this.confirmStatus = null;
        this.rejectStatus = null;
        this.falsePositiveStatus = null;
        this.boxStates = [];
      } finally {
        this.loaded = true;
        this.#inflight = null;
      }
    })();
    return this.#inflight;
  }
}

export const regionStatusesStore = new RegionStatusesStore();
