/**
 * Pure display helpers for `/bakeoff` (v2 comparison wire). Formatting,
 * grouping and request assembly only — every metric, rank, winner and
 * class mapping comes from the server as-is.
 */
import type {
  BakeoffJobState,
  BakeoffMatrix,
  BakeoffModelRef,
  BakeoffRunRequest,
  CustomModelRef,
  EvalDataset,
  FailedTask,
  TrainTestOverlap,
} from '$lib/types_bakeoff';

export const MISSING = '—';

/** Shown for a result the server answers 409 for (not schema v2). */
export const LEGACY_RESULTS_MESSAGE =
  'This run predates the v2 comparison format, so its results cannot be shown here.';

/** Display labels for served metric keys; an unknown key shows verbatim. */
export const METRIC_LABELS: Record<string, string> = {
  map_50: 'mAP@.5',
  map_50_95: 'mAP@.5:.95',
  map_75: 'mAP@.75',
  ap50: 'AP@.5',
  ap50_95: 'AP@.5:.95',
  ap75: 'AP@.75',
  precision: 'Precision',
  recall: 'Recall',
  f1: 'F1',
  mean_iou: 'Mean IoU',
  latency_ms: 'Latency (ms)',
  size_mb: 'Size (MB)',
  coverage: 'Class coverage',
  rank: 'Rank',
};

/** Per-class metric keys `PerClassRow` serves, in display order. */
export const PER_CLASS_METRICS = [
  'ap50_95',
  'ap50',
  'ap75',
  'precision',
  'recall',
  'f1',
] as const;
export type PerClassMetric = (typeof PER_CLASS_METRICS)[number];

export function metricLabel(key: string): string {
  return METRIC_LABELS[key] ?? key;
}

/** Served values are fractions for accuracy and coverage; latency and
 *  size are absolute. Missing (null/undefined/NaN) is "—", never 0. */
export function formatMetric(v: number | null | undefined, key: string): string {
  if (v == null || typeof v !== 'number' || Number.isNaN(v)) return MISSING;
  if (key === 'latency_ms' || key === 'size_mb') return v.toFixed(1);
  if (key === 'rank' || key === 'tp' || key === 'fp' || key === 'fn') return String(v);
  return (v * 100).toFixed(1);
}

/** Caption for the unit of a metric column. */
export function metricUnit(key: string): string {
  if (key === 'latency_ms') return 'milliseconds, lower is better';
  if (key === 'size_mb') return 'megabytes, lower is better';
  return 'percent, higher is better';
}

export function formatCount(v: number | null | undefined): string {
  return v == null ? MISSING : v.toLocaleString();
}

/** Export test splits first (served order), then external datasets grouped
 *  by their served `group`, groups in order of first appearance. */
export function groupEvalDatasets(datasets: EvalDataset[]): {
  exports: EvalDataset[];
  external: { group: string; datasets: EvalDataset[] }[];
} {
  const exports = datasets.filter((d) => d.source === 'export');
  const byGroup = new Map<string, EvalDataset[]>();
  for (const d of datasets) {
    if (d.source !== 'external') continue;
    const g = d.group ?? 'other';
    const list = byGroup.get(g) ?? [];
    list.push(d);
    byGroup.set(g, list);
  }
  return {
    exports,
    external: [...byGroup].map(([group, ds]) => ({ group, datasets: ds })),
  };
}

/** A leakage warning is due when the server reports any overlapping image. */
export function hasOverlap(
  o: TrainTestOverlap | null | undefined,
): o is TrainTestOverlap {
  return o != null && o.n_images > 0;
}

export function formatOverlap(o: TrainTestOverlap): string {
  return `${o.n_images.toLocaleString()} test image${o.n_images === 1 ? '' : 's'} (${(o.fraction * 100).toFixed(1)}%) also in this run's train/val split`;
}

export interface RunSelection {
  datasetIds: string[];
  runIds: string[];
  baselineNames: string[];
  customRefs: CustomModelRef[];
  /** '' = omit, the server's default profile applies. */
  profile: string;
}

export function selectedModelRefs(sel: RunSelection): BakeoffModelRef[] {
  return [
    ...sel.runIds.map((run_id): BakeoffModelRef => ({ source: 'run', run_id })),
    ...sel.baselineNames.map((name): BakeoffModelRef => ({ source: 'baseline', name })),
    ...sel.customRefs,
  ];
}

/** `POST /bakeoff/run` body. The request model forbids extra fields, so
 *  only served keys are sent and unset ones are omitted. */
export function buildRunRequest(sel: RunSelection): BakeoffRunRequest {
  const body: BakeoffRunRequest = {
    datasets: sel.datasetIds.map((id) => ({ id })),
    models: selectedModelRefs(sel),
  };
  if (sel.profile) body.profile = sel.profile;
  return body;
}

/** Custom-model class map: `{"<model class id>": "<eval class name>"}`.
 *  Only JSON shape is checked here; the server resolves names. */
export function parseClassMap(text: string): Record<string, string> | undefined {
  const t = text.trim();
  if (!t) return undefined;
  let v: unknown;
  try {
    v = JSON.parse(t);
  } catch {
    throw new Error('class map is not valid JSON');
  }
  if (!v || typeof v !== 'object' || Array.isArray(v)) {
    throw new Error('class map must be a JSON object');
  }
  for (const [k, name] of Object.entries(v)) {
    if (typeof name !== 'string') {
      throw new Error(`class map entry "${k}" must map to a class name string`);
    }
  }
  return v as Record<string, string>;
}

/** "stage · dataset · model" for whatever a failure names; `job` if none. */
export function failureWhere(f: Pick<FailedTask, 'stage' | 'dataset' | 'model'>): string {
  return [f.stage, f.dataset, f.model].filter(Boolean).join(' · ') || 'job';
}

export function isTerminal(state: BakeoffJobState | null | undefined): boolean {
  return state === 'done' || state === 'error';
}

/** Whether `model` is one of the served winners (ties included). */
export function isBest(
  matrix: Pick<BakeoffMatrix, 'best'> | null,
  datasetId: string,
  metric: string,
  model: string,
): boolean {
  return matrix?.best?.[datasetId]?.[metric]?.includes(model) ?? false;
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return MISSING;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

/**
 * V-5 (fresh-start coordinator review 2026-09-25): the bake-off's mAP and
 * the trainer's own mAP50 differ on the same test split because they use
 * different protocols, and neither page said so. Built from the result's
 * own served `thresholds` (`conf_floor`/`nms_iou` for the mAP sweep,
 * `op_conf`/`op_iou` for the precision/recall/F1 operating point); any
 * other served key is listed verbatim. Null when nothing is served.
 */
export function protocolText(
  thresholds: Record<string, number> | null | undefined,
): string | null {
  if (!thresholds) return null;
  const t = { ...thresholds };
  const parts: string[] = [];
  if (t.conf_floor != null || t.nms_iou != null) {
    const bits: string[] = [];
    if (t.conf_floor != null) bits.push(`conf ≥ ${t.conf_floor}`);
    if (t.nms_iou != null) bits.push(`NMS IoU ${t.nms_iou}`);
    parts.push(`mAP at ${bits.join(', ')}`);
  }
  if (t.op_conf != null || t.op_iou != null) {
    const bits: string[] = [];
    if (t.op_conf != null) bits.push(`conf ${t.op_conf}`);
    if (t.op_iou != null) bits.push(`IoU ${t.op_iou}`);
    parts.push(`precision/recall/F1 at ${bits.join(' · ')}`);
  }
  for (const k of ['conf_floor', 'nms_iou', 'op_conf', 'op_iou']) delete t[k];
  for (const [k, v] of Object.entries(t)) parts.push(`${k} ${v}`);
  return parts.length > 0 ? `bake-off protocol: ${parts.join('; ')}` : null;
}

/** The trainer's own number is Ultralytics' val pass (its defaults, not
 *  a bake-off profile), so it is not directly comparable. */
export const TRAINER_PROTOCOL_LABEL = 'trainer eval (Ultralytics val)';
