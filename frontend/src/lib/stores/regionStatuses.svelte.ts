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

import {
  getRegionStatuses,
  type BoxStateEntry,
  type BoxStateTone,
  type RegionStatusEntry,
} from '$lib/api';
import { onProjectChange } from '$lib/projectChange';

const VALID_TONES: readonly BoxStateTone[] = [
  'accepted',
  'proposed',
  'rejected',
  'neutral',
];

/** Served `tone` → theme color classes, for a served-tone-first box-state
 *  ring/chip (the caller falls back to a role-based client mapping when
 *  `tone` isn't served — see `boxStateTone()` below). One place per output
 *  shape: `MultiBoxCanvas`'s `ringColorFor` wants a CSS `rgb(...)` string;
 *  `SourceImageOverlay`/chip styling want a Tailwind class string. */
const TONE_RING_RGB: Record<BoxStateTone, string> = {
  accepted: 'rgb(74, 222, 128)', // green-400
  proposed: 'rgb(250, 204, 21)', // yellow-400
  rejected: 'rgb(248, 113, 113)', // red-400
  neutral: 'rgb(113, 113, 122)', // zinc-500
};

const TONE_BORDER_CLASS: Record<BoxStateTone, string> = {
  accepted: 'border-green-400',
  proposed: 'border-yellow-400',
  rejected: 'border-red-400',
  neutral: 'border-zinc-500',
};

const TONE_CHIP_CLASS: Record<BoxStateTone, string> = {
  accepted: 'border-green-500/40 bg-green-500/15 text-green-200',
  proposed: 'border-yellow-500/40 bg-yellow-500/15 text-yellow-200',
  rejected: 'border-red-500/40 bg-red-500/15 text-red-200',
  neutral: 'border-zinc-600/40 bg-zinc-700/20 text-zinc-300',
};

/** `rgb(...)` ring color for a served (or, on a pre-tone backend, neutral)
 *  tone. */
export function toneRingRgb(tone: BoxStateTone): string {
  return TONE_RING_RGB[tone];
}

/** Tailwind border-color class for a served (or neutral) tone. */
export function toneBorderClass(tone: BoxStateTone): string {
  return TONE_BORDER_CLASS[tone];
}

/** Tailwind chip classes (border/bg/text) for a served (or neutral) tone. */
export function toneChipClass(tone: BoxStateTone): string {
  return TONE_CHIP_CLASS[tone];
}

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
  #gen = 0;

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

  /** The served `tone` for a box `state` value — `'neutral'` when
   *  `box_states` hasn't loaded, the state is unrecognized, or the
   *  backend predates `tone` (a pre-W8.7-follow-up backend). Callers use
   *  this (never the raw `role`/`state` string) to pick a ring/chip
   *  color, so a served tone always wins over any client role→color
   *  guess. */
  boxStateTone(value: string | null | undefined): BoxStateTone {
    const tone = this.boxStateInfo(value)?.tone;
    return tone && VALID_TONES.includes(tone) ? tone : 'neutral';
  }

  async init(): Promise<void> {
    if (this.loaded) return;
    if (this.#inflight) return this.#inflight;
    const gen = this.#gen;
    this.#inflight = (async () => {
      try {
        const res = await getRegionStatuses();
        // A load started for the previous project never lands.
        if (gen !== this.#gen) return;
        this.list = res.statuses ?? [];
        this.confirmStatus = res.confirm_status ?? null;
        this.rejectStatus = res.reject_status ?? null;
        this.falsePositiveStatus = res.false_positive_status ?? null;
        this.boxStates = res.box_states ?? [];
      } catch {
        if (gen !== this.#gen) return;
        this.list = [];
        this.confirmStatus = null;
        this.rejectStatus = null;
        this.falsePositiveStatus = null;
        this.boxStates = [];
      } finally {
        if (gen === this.#gen) {
          this.loaded = true;
          this.#inflight = null;
        }
      }
    })();
    return this.#inflight;
  }

  /** Project switch: the vocabulary is per project. */
  resetForProjectChange(): void {
    this.#gen += 1;
    this.#inflight = null;
    this.list = [];
    this.confirmStatus = null;
    this.rejectStatus = null;
    this.falsePositiveStatus = null;
    this.boxStates = [];
    this.loaded = false;
  }
}

export const regionStatusesStore = new RegionStatusesStore();
onProjectChange(() => regionStatusesStore.resetForProjectChange());
