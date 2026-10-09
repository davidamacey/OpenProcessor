/**
 * Promote-to-Triton helpers (F-64, fresh-start findings 2026-09-25).
 *
 * `POST {API_PREFIX}/train/promote/{job_id}` answers every promote-blocking
 * failure with a 422 whose `detail` is `PromoteGateFailedDetail`:
 * `{message, failures: [{code, message, class_name?}], force_allowed,
 * override?, thresholds?}`. The modal renders it as served; whether a
 * force retry is offered is the server's `force_allowed`, never a client
 * guess.
 */
import { ApiError } from '$lib/api';
import { isPromoteActive } from '$lib/promoteJobController.svelte';
import type { PromoteJobStatus, PromoteResponse, TrainJobStatus } from '$lib/types_train';

export interface PromoteGateFailure {
  code: string;
  message: string;
  class_name?: string | null;
}

export interface PromoteGateDetail {
  message: string;
  failures: PromoteGateFailure[];
  force_allowed: boolean;
  override?: string | null;
  thresholds?: Record<string, number> | null;
}

export function promoteGateDetail(e: unknown): PromoteGateDetail | null {
  if (!(e instanceof ApiError) || e.status !== 422) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object' || Array.isArray(detail)) return null;
  const d = detail as Record<string, unknown>;
  if (typeof d.message !== 'string' || !Array.isArray(d.failures)) return null;
  const failures: PromoteGateFailure[] = [];
  for (const raw of d.failures) {
    if (!raw || typeof raw !== 'object') continue;
    const f = raw as Record<string, unknown>;
    if (typeof f.code !== 'string' || typeof f.message !== 'string') continue;
    failures.push({
      code: f.code,
      message: f.message,
      class_name: typeof f.class_name === 'string' ? f.class_name : null,
    });
  }
  const thresholds =
    d.thresholds && typeof d.thresholds === 'object'
      ? (d.thresholds as Record<string, number>)
      : null;
  return {
    message: d.message,
    failures,
    force_allowed: d.force_allowed === true,
    override: typeof d.override === 'string' ? d.override : null,
    thresholds,
  };
}

/** The served message sometimes already starts with "<class_name>:"; the
 *  modal adds the class itself, so strip it to avoid "object:object: …". */
export function gateFailureMessage(f: PromoteGateFailure): string {
  const name = f.class_name;
  if (!name) return f.message;
  const prefix = `${name}:`;
  return f.message.startsWith(prefix)
    ? f.message.slice(prefix.length).trimStart()
    : f.message;
}

/** `PromoteRequest.triton_name` limits: `[A-Za-z0-9_-]`, 1..64 chars. */
const TRITON_NAME_MAX = 64;

/**
 * A generic default Triton model name for a run: its own job id made
 * Triton-safe (the job id already carries the timestamp and model size).
 * No version suffix: nothing here knows the deployment's versioning.
 */
export function defaultTritonName(jobId: string): string {
  const safe = jobId
    .replace(/[^A-Za-z0-9_-]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '');
  return safe.slice(0, TRITON_NAME_MAX) || 'model';
}

/** Success toast after a promote. Adds the slow-first-prediction note only
 *  when the server says a cold start is expected (OpenProcessor ffb88b8). */
export function promoteSuccessMessage(
  res: Pick<PromoteResponse, 'triton_name' | 'cold_start_expected_on_first_inference'>,
): string {
  const base = `Promoted ${res.triton_name} → Triton`;
  return res.cold_start_expected_on_first_inference
    ? `${base}. The first prediction will be slow while its engine builds.`
    : base;
}

/**
 * Page-reload resume: the first of `runJobIds` whose served train status
 * (`GET /train/status/{job_id}`, the only route carrying `promote`) has a
 * promote still in flight. A failed lookup skips that run.
 */
export async function findActivePromote(
  runJobIds: string[],
  getStatus: (jobId?: string) => Promise<TrainJobStatus | null>,
): Promise<{ runJobId: string; promote: PromoteJobStatus } | null> {
  for (const runJobId of runJobIds) {
    try {
      const st = await getStatus(runJobId);
      if (st?.promote && isPromoteActive(st.promote)) {
        return { runJobId, promote: st.promote };
      }
    } catch {
      // Non-fatal: resume is best effort.
    }
  }
  return null;
}
