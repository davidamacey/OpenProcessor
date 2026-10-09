/**
 * Pure helpers for the `/settings` "Curation scores" card
 * (docs/design/frontend-coverage-audit-2026-09-24.md §G10). Split out of
 * `ScoresCard.svelte` so the null→"—" formatting and the compute-job
 * status classification are unit-testable without mounting a component,
 * same convention as `trainResults.ts`'s `formatMetric`/`formatScalar`
 * and `embeddingPlot.ts`'s `classifyRebuildPoll`.
 */

import type { ScoreCoverageEntry, ScoresJob } from './api';

/** "N / total", or "—" when the scorer has no coverage row at all
 *  (never a bare `0`, matching `formatCount()`'s null-vs-zero rule). */
export function formatCoverageCounts(entry: ScoreCoverageEntry | undefined): string {
  if (!entry) return '—';
  return `${entry.n_scored.toLocaleString()} / ${entry.total.toLocaleString()}`;
}

/** "NN%", or "—" when the scorer has no coverage row at all. */
export function formatCoveragePct(entry: ScoreCoverageEntry | undefined): string {
  if (!entry) return '—';
  return `${entry.pct}%`;
}

export type ScoresPollOutcome = 'running' | 'completed' | 'failed' | 'cancelled';

/** What a poll of `GET /scores/status` should do next, given only the
 *  job's own `status` — the compute job is a client-initiated singleton
 *  (the card only ever polls while it itself started or adopted a run),
 *  so unlike `classifyRebuildPoll` there's no "was I even tracking this"
 *  ambiguity to resolve. */
export function classifyScoresPoll(status: ScoresJob['status']): ScoresPollOutcome {
  if (status === 'running') return 'running';
  if (status === 'failed') return 'failed';
  if (status === 'cancelled') return 'cancelled';
  // 'completed' and the terminal-but-never-started 'idle' both mean
  // "stop polling, nothing more to report as a failure" — 'idle' should
  // be unreachable here (the card only polls after seeing 'running'),
  // but a job that raced to completion between two polls must still
  // resolve as success, not silently do nothing.
  return 'completed';
}
