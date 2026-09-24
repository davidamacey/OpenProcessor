/**
 * Pure logic for the `/models` page's unload action (follow-up gap 2,
 * docs/design/audit-remediation-plan-2026-09.md Appendix D item 3,
 * 2026-09-11 — "No way to unload/remove a promoted Triton model from the
 * UI"). Extracted from the Svelte component (mirroring `trainGpuOptions.ts`'s
 * pattern) so the guard behavior is directly unit-testable — this repo has
 * no `@testing-library/svelte` (see `StrategyBar.test.ts`).
 *
 * The real guard lives server-side (`DELETE /curation/models/{name}` in
 * openprocessor's `op_models.py`) — this module only decides what the button
 * *looks like* from the `is_region_protected` / `requires_force_to_unload` flags the
 * server already computed and sent back on `/curation/models/status`. It must
 * never invent its own notion of "is this LPR / is this active" — that
 * would be a second, driftable copy of the real guard.
 */

import type { OpModel } from './types';

export type UnloadButtonState = 'hidden' | 'normal' | 'force-required';

/**
 * - `hidden`: never rendered — non-Triton models (Gemma) and any LPR
 *   model (the one guard with no override, per CLAUDE.md's "never touch
 *   any LPR model or LPR data" constraint).
 * - `force-required`: rendered, but the action requires an explicit
 *   second, stronger confirmation and is sent with `force=true` — the
 *   active vehicle model or another core pipeline model currently
 *   serving live traffic.
 * - `normal`: rendered, single confirmation, `force=false`.
 */
export function unloadButtonState(
  model: Pick<OpModel, 'kind' | 'is_region_protected' | 'requires_force_to_unload'>,
): UnloadButtonState {
  if (model.kind !== 'triton') return 'hidden';
  if (model.is_region_protected) return 'hidden';
  if (model.requires_force_to_unload) return 'force-required';
  return 'normal';
}

export function unloadConfirmMessage(
  model: Pick<OpModel, 'name' | 'requires_force_to_unload'>,
): string {
  if (model.requires_force_to_unload) {
    return (
      `${model.name} is currently serving live traffic (the active vehicle model, or another ` +
      'core pipeline model). Unloading it removes it from Triton and permanently deletes its ' +
      'files on disk — inference for this model will be DOWN until a replacement is loaded. ' +
      'This cannot be undone. Continue?'
    );
  }
  return (
    `Unload ${model.name}? This removes it from Triton and permanently deletes its files on ` +
    'disk. This cannot be undone.'
  );
}

/** Second confirmation shown only for `force-required` models. */
export function unloadForceConfirmMessage(model: Pick<OpModel, 'name'>): string {
  return `Really force-unload ${model.name}? This is your last chance to back out.`;
}
