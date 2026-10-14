/** Display formatting for served ingest-source status. No decisions: every
 *  timestamp and count shown is the one the backend served. */
import type { SourceScanStatus } from '$lib/types_ingestSources';

/** An ISO timestamp (or epoch seconds) in the viewer's locale; `--` when absent. */
export function formatWhen(value: string | number | null | undefined): string {
  if (value == null || value === '') return '--';
  const date = typeof value === 'number' ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) return '--';
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function scanText(scan: SourceScanStatus): string {
  return `scanning: ${scan.pages} pages, ${scan.listed} listed, ${scan.queued} queued`;
}

export const STATUS_TONE: Record<string, string> = {
  queued: 'text-zinc-300',
  running: 'text-blue-300',
  completed: 'text-emerald-300',
  completed_with_errors: 'text-amber-300',
  failed: 'text-red-300',
  cancelled: 'text-amber-300',
  interrupted: 'text-amber-300',
};
