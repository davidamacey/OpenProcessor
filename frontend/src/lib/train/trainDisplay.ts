/**
 * /train display helpers: pure string mappings from served values to
 * labels and pill classes. No state, no fetches.
 */
import type { ExportDataset } from '$lib/types_export';
import type { PreflightCheck, TrainJobStatus } from '$lib/types_train';

export function statePillClass(s: string): string {
  switch (s) {
    case 'running':
    case 'starting':
      return 'bg-accent/20 text-accent border-accent/40';
    case 'exporting':
      return 'bg-holdout/20 text-holdout border-holdout/40';
    case 'finished':
      return 'bg-success/20 text-success border-success/40';
    case 'failed':
      return 'bg-danger/20 text-danger border-danger/40';
    case 'cancelled':
    case 'skipped':
      return 'bg-surface-3 text-fg-2 border-line-strong';
    case 'lost':
      return 'bg-warn/20 text-warn border-warn/40';
    case 'queued':
    default:
      return 'bg-surface-2 text-fg-2 border-line-strong';
  }
}

export function familyLabel(_r: TrainJobStatus): string {
  // The status doc doesn't carry model_family; the trainer guarantees
  // YOLO26 today. Keep the column ready for future families though.
  return 'yolo26';
}

// Human-readable label for a dataset option in the picker.
export function datasetLabel(d: ExportDataset): string {
  const n = d.image_count != null ? d.image_count.toLocaleString() : '?';
  const tag = d.version_tag ? ` · ${d.version_tag}` : '';
  const samp = d.sampling === 'stratified_even' ? ' · sampled' : '';
  const cur = d.is_current ? ' · current' : '';
  const when = d.exported_at ? d.exported_at.slice(0, 16).replace('T', ' ') : '';
  const dir = d.export_dir.split('/').pop() ?? d.export_dir;
  return `${dir} (${n} imgs${samp}${tag}${cur}) ${when}`.trim();
}

export function checkSeverityClass(s: PreflightCheck['severity']): string {
  switch (s) {
    case 'block':
      return 'border-danger/40 bg-danger/10 text-danger';
    case 'warn':
      return 'border-warn/40 bg-warn/10 text-warn';
    default:
      return 'border-line-strong bg-surface-0 text-fg-2';
  }
}
