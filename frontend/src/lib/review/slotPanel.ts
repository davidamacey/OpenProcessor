/**
 * Pure data-returning helpers for `/review`'s inline slot panel body
 * (Finding D, docs/design/slot-generic-crop-mapping-plan-2026-09-21.md
 * §6.2). Same shape as `slotKeymap.ts`: no side effects, separately
 * unit tested, each function's doc comment names the hand-maintained
 * copy it replaces.
 *
 * These generalize the panel's STATUS VOCABULARY and COPY, not its
 * field reads — `SlotData` (from `readSlot`, via `slotOf`) is what the
 * page reads for the actual values.
 */

import type { SlotSpec } from '../annotations/types';
import type { RegionStatusEntry } from '../api';
import { keymapStore } from '$stores/keymap.svelte';

/**
 * Human-writable lifecycle states, in server-declared order when the
 * deployment's `GET {API_PREFIX}/regions/statuses` vocabulary (`served`)
 * is available — falling back to the profile's own declared
 * `capabilities.lifecycle.states` only when it isn't (endpoint absent,
 * or `regionStatusesStore` hasn't loaded yet).
 */
export function humanWritableStates(
  spec: SlotSpec,
  served?: readonly RegionStatusEntry[] | null,
): Array<{ value: string; label: string }> {
  if (served && served.length > 0) {
    return served
      .filter((s) => s.human_writable)
      .map((s) => ({ value: s.value, label: s.label }));
  }
  return (spec.capabilities.lifecycle?.states ?? [])
    .filter((s) => s.humanWritable)
    .map((s) => ({ value: s.value, label: s.label }));
}

/**
 * True when the rejection-reason input should render for `status`.
 * Reads the server's own `wants_reason` (+ `human_writable`) flags when
 * `served` is available; otherwise falls back to the profile's role ===
 * 'rejected' | 'absent' heuristic.
 *
 * For a slot declaring the standard region lifecycle the fallback yields
 * exactly {verify_rejected, no_region_visible} — pinned in
 * slotPanel.test.ts.
 */
export function statusWantsRejectionReason(
  spec: SlotSpec,
  status: string,
  served?: readonly RegionStatusEntry[] | null,
): boolean {
  if (served && served.length > 0) {
    const entry = served.find((s) => s.value === status);
    return !!entry && entry.human_writable && entry.wants_reason;
  }
  const state = spec.capabilities.lifecycle?.states.find((s) => s.value === status);
  if (!state || !state.humanWritable) return false;
  return state.role === 'rejected' || state.role === 'absent';
}

export interface SlotPanelLabels {
  scoreLabel: string;
  statusLabel: string;
  textLabel: string;
  textPlaceholder: string;
  confirmLabel: string;
  rejectLabel: string;
  noBoxHint: string;
}

/** Field labels for the panel's <dl>, from label.* + capabilities.text. */
export function panelLabels(spec: SlotSpec): SlotPanelLabels {
  return {
    scoreLabel: `${spec.label.title} score`,
    statusLabel: `${spec.label.title} status`,
    textLabel: spec.capabilities.text?.label ?? `${spec.label.title} text`,
    textPlaceholder: spec.capabilities.text?.placeholder ?? '',
    confirmLabel: `Confirm ${spec.label.title}`,
    rejectLabel: `Reject (no ${spec.label.singular})`,
    noBoxHint: `No ${spec.label.singular} bbox on this crop — press ${keymapStore.glyph('review.region.edit_box')} to draw one.`,
  };
}
