/**
 * Pure logic for the `/models` page's unload action (follow-up gap 2,
 * docs/design/audit-remediation-plan-2026-09.md Appendix D item 3,
 * 2026-09-11 — "No way to unload/remove a promoted Triton model from the
 * UI"). Extracted from the Svelte component (mirroring the extracted-helper
 * pattern) so the guard behavior is directly unit-testable — this repo has
 * no `@testing-library/svelte` (see `StrategyBar.test.ts`).
 *
 * The real guard lives server-side (`DELETE {API_PREFIX}/models/{name}` in
 * OpenProcessor's `models.py`) — this module only decides what the button
 * *looks like* from the `is_region_protected` / `requires_force_to_unload` flags the
 * server already computed and sent back on `{API_PREFIX}/models/status`. It must
 * never invent its own notion of "is this protected / is this active" — that
 * would be a second, driftable copy of the real guard.
 */

import type { ModelInfo } from './types';

export type UnloadButtonState = 'hidden' | 'normal' | 'force-required';

/**
 * - `hidden`: never rendered — either the server says outright this
 *   entry can't be unloaded at all (`unloadable === false`, e.g. the
 *   external segmenter/VLM entries, 2026-09-25 follow-up to #36 item 5),
 *   or it's region-protected (`is_region_protected`, the one guard with
 *   no override: region models and their data are never touched from
 *   here — as of 698d1da this also covers the ingest primary proposer/
 *   secondary classifier and the OCR det/rec pair, not just the region
 *   detector).
 * - `force-required`: rendered, but the action requires an explicit
 *   second, stronger confirmation and is sent with `force=true` — the
 *   active production model or another core pipeline model currently
 *   serving live traffic.
 * - `normal`: rendered, single confirmation, `force=false`.
 *
 * `unloadable` is checked FIRST and is the server's own verdict — never
 * re-derived from `kind`. A backend that predates the field (`unloadable`
 * absent/`undefined`) falls back to the prior `kind !== 'triton'` rule,
 * so an older deployment renders exactly as before.
 */
export function unloadButtonState(
  model: Pick<
    ModelInfo,
    'kind' | 'is_region_protected' | 'requires_force_to_unload' | 'unloadable'
  >,
): UnloadButtonState {
  if (model.unloadable === false) return 'hidden';
  if (model.unloadable === undefined && model.kind !== 'triton') return 'hidden';
  if (model.is_region_protected) return 'hidden';
  if (model.requires_force_to_unload) return 'force-required';
  return 'normal';
}

/** Whether to render the "protected: in use by the pipeline" chip next
 *  to an unloadable-but-region-protected model — distinct from a model
 *  that's simply not unloadable at all (an external service), which gets
 *  no chip and no button. */
export function showsProtectedChip(
  model: Pick<ModelInfo, 'is_region_protected' | 'unloadable'>,
): boolean {
  return !!model.is_region_protected && model.unloadable !== false;
}

export function unloadConfirmMessage(
  model: Pick<ModelInfo, 'name' | 'requires_force_to_unload'>,
): string {
  if (model.requires_force_to_unload) {
    return (
      `${model.name} is currently serving live traffic (the active production model, or another ` +
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
export function unloadForceConfirmMessage(model: Pick<ModelInfo, 'name'>): string {
  return `Really force-unload ${model.name}? This is your last chance to back out.`;
}
