/**
 * The whole-project auto-label job (`/pipeline/auto_label/*`). A VLM run on a
 * cluster, sub-cluster, selection, box or crop is an assist run (`api_assist.ts`).
 */
import { apiFetch, qs, scoped } from '$lib/api';
import { ApiError } from '$lib/api_errors';

/** The 422 `detail` an unknown per-run strategy id (e.g. `prompt_pack`)
 *  produces on `auto_label/start`. */
export interface UnknownStrategyDetail {
  axis: string;
  requested: string;
  valid_ids: string[];
}

export function unknownStrategyDetail(e: unknown): UnknownStrategyDetail | null {
  if (!(e instanceof ApiError) || e.status !== 422) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object') return null;
  const d = detail as Record<string, unknown>;
  if (typeof d.axis !== 'string' || typeof d.requested !== 'string') return null;
  if (!Array.isArray(d.valid_ids)) return null;
  return {
    axis: d.axis,
    requested: d.requested,
    valid_ids: d.valid_ids.filter((v): v is string => typeof v === 'string'),
  };
}

// -- Auto-label (recluster) job ------------------------------------------
// Wraps POST {API_PREFIX}/pipeline/auto_label/{start,status,cancel}. The pipeline
// re-runs prototype assignment → cluster_id normalize → AHC residuals → auto-
// promote → VLM sweep, fixing prototype drift and stale cluster_id on
// labeled crops. Hours at HDD scale; the panel polls status while it runs.

export interface AutoLabelStartParams {
  train_clusters?: boolean;
  promote_min_purity?: number;
  promote_min_members?: number;
  vlm_batch_size?: number;
  vlm_concurrency?: number;
  max_vlm_crops?: number;
  classifier_confidence_skip_vlm?: number;
  /** When true, broaden the residual AHC pool to include items already
   *  in candidate clusters so smaller candidates can merge into bigger
   *  ones. Default false — only fresh / class-bucketed items are
   *  re-clustered. */
  recluster_unvalidated?: boolean;
  /** v0.4.0: embed the items that have no vector as the run's first stage.
   *  Sent only when ticked. */
  embed_missing?: boolean;
  // -- Cluster scope (primary-subject gate) ------------------------------
  /** Train + assign only crops with crop_rank_in_image <= this (1 = largest,
   *  2 = largest + 2nd). Smaller crops are parked. Needs full rank/blur
   *  backfill; the run blocks otherwise. */
  gate_max_rank?: number | null;
  /** Train + assign only crops with blur_lap_ratio >= this. */
  gate_min_blur_ratio?: number | null;
  /** IVF fixed centroid count (default 512). Sweep down with the gate on. */
  n_clusters?: number | null;
  // -- VLM-assisted scoping (2026-09-20 contract, this plan §1.3) --------
  /**
   * Scope the run to a single class instead of the whole pool — "just
   * help me with forklifts right now". Same `class_id` filter convention
   * as `getCrops`/`getClusters`/`/select/diverse` elsewhere in this
   * file. `null`/omitted = today's unscoped, whole-dataset behavior, and
   * `qs()` drops it entirely so an unscoped request stays byte-identical
   * to every request this app has ever sent.
   *
   * Accepted by OpenProcessor `main`'s `auto_label/start`. The UI never
   * sends it unless `/methods` advertises the assist axes — see
   * `isScopedAssistAvailable` in `$lib/strategies`.
   */
  class_id?: number | null;
  /**
   * Per-run override of the settings-doc prompt-pack default, for this
   * job only — omitted/null means the deployment default. An unknown id
   * is a 422 (`unknownStrategyDetail`); the resolved value is echoed in
   * the job state's `args`. Produced in exactly one place
   * (`createAssistScope().toStartParams()` in `$lib/assistScope.svelte`).
   */
  prompt_pack?: string | null;
  /**
   * per-run VLM endpoint (a registry name, or `off`) for this job
   * only; omitted/null = the project's active endpoint. An unknown id is
   * a 422 `unknown_vlm`. Produced in exactly one place
   * (`createAssistScope().toStartParams()`).
   */
  vlm?: string | null;
  /**
   * acknowledges that this run's crops go to an external endpoint.
   * Sent only when `true` (`startAutoLabel` drops `false`).
   */
  acknowledge_external?: boolean;
  /**
   * G5: `pipeline.py`'s `run_vlm: bool = Query(False)` — the VLM sweep
   * stage is opt-in server-side and defaults off. Previously never sent
   * at all, so a scoped run (a class or pack picked via AssistScopeBar)
   * silently skipped the VLM stage it claimed to scope
   * (`result.stages.vlm.skipped=true` live). Omitted/undefined keeps the
   * request byte-identical to an unscoped run; the caller decides when
   * to send `true`.
   */
  run_vlm?: boolean;
}

export type AutoLabelStatus = 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface AutoLabelJobState {
  job_id: string;
  status: AutoLabelStatus;
  stage: string;
  processed: number;
  total: number;
  started_at: number;
  finished_at: number;
  error: string | null;
  result: Record<string, unknown>;
  args: Record<string, unknown>;
  eta_seconds: number | null;
  elapsed_seconds: number;
  // GPU clustering telemetry (optional — older worker versions omit).
  // `backend` is 'gpu' when cluster_residuals' UMAP ran on cuML;
  // 'cpu' for sklearn fallback. `backend_detail` is a human-readable
  // chip ("gpu (cuml 26.4 · A6000 GPU 0 · 9824/49152 MB free · nn_descent)"
  // or "cpu (sklearn 1.4 · umap-learn 0.5)").
  backend?: 'gpu' | 'cpu' | null;
  backend_detail?: string | null;
  free_vram_mb?: number | null;
  peak_vram_mb?: number | null;
  stage_durations?: Record<string, number>;
}

export function startAutoLabel(
  params: AutoLabelStartParams = {},
  signal?: AbortSignal,
): Promise<AutoLabelJobState> {
  const { acknowledge_external: ack, ...rest } = params;
  const sent = ack === true ? { ...rest, acknowledge_external: true } : rest;
  return apiFetch<AutoLabelJobState>(
    `${scoped()}/pipeline/auto_label/start${qs(sent as Record<string, unknown>)}`,
    { method: 'POST' },
    signal,
  );
}

export function getAutoLabelStatus(signal?: AbortSignal): Promise<AutoLabelJobState> {
  return apiFetch<AutoLabelJobState>(
    `${scoped()}/pipeline/auto_label/status`,
    {},
    signal,
  );
}

export function cancelAutoLabel(
  signal?: AbortSignal,
): Promise<AutoLabelJobState & { cancelled: boolean }> {
  return apiFetch<AutoLabelJobState & { cancelled: boolean }>(
    `${scoped()}/pipeline/auto_label/cancel`,
    { method: 'POST' },
    signal,
  );
}
