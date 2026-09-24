/**
 * G5: whether AutoLabelPanel's start() should send `run_vlm: true`.
 * Factored out of the component so the decision is unit-testable
 * without mounting Svelte — same rationale as `resolveStatsUpdate`.
 *
 * `pipeline.py`'s `run_vlm` defaults to False server-side, and a scoped
 * run (a class or prompt pack picked via AssistScopeBar) is a no-op
 * without it — the VLM sweep is the only stage `class_id`/`prompt_pack`
 * actually scope. Live: an unscoped `run_vlm` was never sent at all, so
 * `args.run_vlm=false` and `result.stages.vlm.skipped=true` even on a
 * run that claimed "VLM labeling limited to {class}".
 */
export function resolveAutoLabelRunVlm(
  scopeParams: Record<string, unknown>,
  checkboxChecked: boolean,
): boolean {
  return checkboxChecked || Object.keys(scopeParams).length > 0;
}
