/**
 * The 409 `near_dup_stale` refusal of the holdout freeze and the training
 * exports: `{detail: {code: 'near_dup_stale', levels: [{level, stale_reasons,
 * pending_count}]}}`. The server raises it only where a holdout is or would be
 * frozen, so the UI shows it only when it arrives and never pre-checks.
 */
import { ApiError } from '$lib/api_errors';
import type {
  NearDupRunMode,
  NearDupStaleLevel,
  NearDupStaleReason,
} from '$lib/types_nearDup';

/** The stale levels of a `near_dup_stale` refusal, or null for any other error. */
export function nearDupStaleLevels(e: unknown): NearDupStaleLevel[] | null {
  if (!(e instanceof ApiError) || e.status !== 409) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object' || Array.isArray(detail)) return null;
  const d = detail as { code?: unknown; levels?: unknown };
  if (d.code !== 'near_dup_stale' || !Array.isArray(d.levels)) return null;
  return d.levels as NearDupStaleLevel[];
}

const REASON_TEXT: Record<NearDupStaleReason, string> = {
  pending: 'has items not yet grouped',
  rebuild_required: 'needs a full rebuild',
  no_run: 'has never been built',
};

export function staleReasonText(reason: NearDupStaleReason): string {
  return REASON_TEXT[reason];
}

/** The run that clears a level's staleness: a full build unless only new items are pending. */
export function runModeFor(level: NearDupStaleLevel): NearDupRunMode {
  return level.stale_reasons.length === 1 && level.stale_reasons[0] === 'pending'
    ? 'incremental'
    : 'full';
}
