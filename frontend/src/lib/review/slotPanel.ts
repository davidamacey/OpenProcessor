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

/**
 * Human-writable lifecycle states, in the profile's declared order.
 * Replaces review/+page.svelte's PLATE_STATUS_OPTIONS, which read a
 * directly-imported licensePlateSlot regardless of the active tab.
 */
export function humanWritableStates(
  spec: SlotSpec,
): Array<{ value: string; label: string }> {
  return (spec.capabilities.lifecycle?.states ?? [])
    .filter((s) => s.humanWritable)
    .map((s) => ({ value: s.value, label: s.label }));
}

/**
 * True when writing `status` means the sub-box must be cleared — i.e.
 * status === lifecycle.rejectState. Replaces the hardcoded
 * `editedPlateStatus === 'no_plate_visible'` check.
 */
export function statusClearsBox(spec: SlotSpec, status: string): boolean {
  return status.length > 0 && status === spec.capabilities.lifecycle?.rejectState;
}

/**
 * True when the rejection-reason input should render for `status`: the
 * state's role is 'rejected' or 'absent' *and* it is humanWritable.
 * Replaces the hardcoded
 * `=== 'verify_rejected' || === 'no_plate_visible'` check.
 *
 * For licensePlateSlot this yields exactly {verify_rejected,
 * no_plate_visible} — pinned in slotPanel.test.ts as the no-regression
 * proof.
 */
export function statusWantsRejectionReason(spec: SlotSpec, status: string): boolean {
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
    noBoxHint: `No ${spec.label.singular} bbox on this crop — press E to draw one.`,
  };
}
