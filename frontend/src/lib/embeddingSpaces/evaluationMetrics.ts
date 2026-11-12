/**
 * An evaluation's `metrics` and `sample_spec` are documents the server stores as written (recall@k,
 * MRR, intervals, per stratum, seed, n ...). They are shown as served: one row per leaf, its path
 * dotted. Nothing is computed, ranked or compared here.
 */
export interface MetricRow {
  path: string;
  value: string;
}

function leaf(v: unknown): string {
  if (typeof v === 'number')
    return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(4)));
  if (v === null) return 'null';
  if (Array.isArray(v)) return v.map(leaf).join(', ');
  return String(v);
}

export function metricRows(doc: Record<string, unknown>, prefix = ''): MetricRow[] {
  return Object.entries(doc).flatMap(([key, v]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return v !== null && typeof v === 'object' && !Array.isArray(v)
      ? metricRows(v as Record<string, unknown>, path)
      : [{ path, value: leaf(v) }];
  });
}
