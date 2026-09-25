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

import { getRegionStatuses, type RegionStatusEntry } from '$lib/api';

class RegionStatusesStore {
  list = $state<RegionStatusEntry[]>([]);
  confirmStatus = $state<string | null>(null);
  rejectStatus = $state<string | null>(null);
  falsePositiveStatus = $state<string | null>(null);
  loaded = $state<boolean>(false);
  #inflight: Promise<void> | null = null;

  /** The served label for a stored status value, or `null` when the
   *  vocabulary isn't loaded or doesn't know the value (callers fall back
   *  to the slot profile's own label). */
  labelFor(value: string | null | undefined): string | null {
    if (!value) return null;
    return this.list.find((s) => s.value === value)?.label ?? null;
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
      } catch {
        this.list = [];
        this.confirmStatus = null;
        this.rejectStatus = null;
        this.falsePositiveStatus = null;
      } finally {
        this.loaded = true;
        this.#inflight = null;
      }
    })();
    return this.#inflight;
  }
}

export const regionStatusesStore = new RegionStatusesStore();
