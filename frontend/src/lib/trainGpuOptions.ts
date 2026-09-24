/**
 * GPU picker options for the `/train` form (follow-up gap 1,
 * docs/design/audit-remediation-plan-2026-09.md Appendix D item 2,
 * 2026-09-11 — "No single-GPU picker in the `/train` UI").
 *
 * The form previously only offered "both A6000s" or "GPU 0 only" —
 * pinning a run to GPU 2 alone required an undocumented
 * `hyperparameters.device` override on the backend, discovered only by
 * reading `docker/trainer/` directly during Phase 10's live
 * verification. The backend (`TrainJobSpec.cuda_visible_devices`, plus
 * the trainer's host→container-local GPU index mapping and the arbiter's
 * `needs_gemma_stop`) now formally supports and correctly handles a
 * lone `'2'` claim; this module is the UI-side single source of truth
 * for which host GPUs are ever offered.
 *
 * This host has 3 GPUs: slot 0 and slot 2 are A6000s (the only ones the
 * legacy-trainer container is ever attached to — see
 * `docker-compose.legacy.yml`'s `device_ids: ["0", "2"]`); slot 1 is a
 * 3080 Ti dedicated to an unrelated app and must NEVER be offered here,
 * full stop — extracted into its own array (rather than left inline in
 * `TrainForm.svelte`) specifically so this exhaustive list is directly
 * unit-testable without a component-mount harness (this repo has no
 * `@testing-library/svelte`, per `StrategyBar.test.ts`'s convention).
 */

export interface GpuOption {
  /** The exact `cuda_visible_devices` value sent on `POST {API_PREFIX}/train/start`. */
  value: string;
  label: string;
  /** True when this claim leaves one A6000 available to Gemma/SAM3. */
  warn: boolean;
}

/**
 * The only 3 selectable GPU claims. This array is the exhaustive
 * allowlist — the host GPU ids referenced across every entry's `value`
 * must never include `1` (3080 Ti). Keep in sync with the backend's
 * allowlist (`src/services/legacy/train_jobs.py`'s
 * `_ALLOWED_TRAIN_GPU_IDS = {0, 2}` in openprocessor), which independently
 * enforces the same constraint server-side as defense in depth.
 */
export const GPU_OPTIONS: readonly GpuOption[] = [
  { value: '0,2', label: '2× A6000 (slots 0, 2)', warn: false },
  { value: '0', label: '1× A6000 (slot 0)', warn: true },
  { value: '2', label: '1× A6000 (slot 2)', warn: true },
] as const;

/** The one GPU id that must never appear in any `GpuOption.value`. */
export const FORBIDDEN_GPU_ID = '1';

/**
 * Inline advisory text shown under the GPU picker, describing what
 * happens to Gemma/SAM3 for the selected claim. `null` = no advisory
 * (the default dual-GPU claim already describes itself via its label).
 */
export function gpuAdvisory(cudaDevices: string): string | null {
  switch (cudaDevices) {
    case '0':
      return 'Gemma worker stays alive on GPU 2.';
    case '2':
      // Gemma's vLLM server lives on GPU 2 — the arbiter stops it
      // entirely for a GPU-2 claim (not just a worker pause) so the
      // trainer isn't contending with it on the same physical GPU.
      return "Gemma's model server on GPU 2 is stopped for this run's duration (frees VRAM for training) and restarts automatically when the run finishes.";
    case '0,2':
      return 'Claims both A6000s — SAM3 and Gemma are stopped for the duration of the run and restart automatically afterward.';
    default:
      return null;
  }
}
