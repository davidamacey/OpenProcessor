/**
 * /train display helpers: pure string mappings from served values to
 * labels and pill classes. No state, no fetches.
 */
import type { ExportDataset } from '$lib/types';
import type { PreflightCheck, TrainJobStatus } from '$lib/types_train';

export function statePillClass(s: string): string {
  switch (s) {
    case 'running':
    case 'starting':
      return 'bg-blue-500/20 text-blue-200 border-blue-500/40';
    case 'exporting':
      return 'bg-purple-500/20 text-purple-200 border-purple-500/40';
    case 'finished':
      return 'bg-green-500/20 text-green-200 border-green-500/40';
    case 'failed':
      return 'bg-red-500/20 text-red-200 border-red-500/40';
    case 'cancelled':
    case 'skipped':
      return 'bg-zinc-700 text-zinc-300 border-zinc-600';
    case 'lost':
      return 'bg-yellow-500/20 text-yellow-200 border-yellow-500/40';
    case 'queued':
    default:
      return 'bg-zinc-800 text-zinc-300 border-zinc-700';
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
      return 'border-red-500/40 bg-red-500/10 text-red-200';
    case 'warn':
      return 'border-yellow-500/40 bg-yellow-500/10 text-yellow-200';
    default:
      return 'border-zinc-700 bg-zinc-950 text-zinc-300';
  }
}
