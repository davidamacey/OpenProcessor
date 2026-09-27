/**
 * Compact per-file result storage for an ingest run
 * (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §A.2). A
 * `Map<id, IngestFileResult>` plus running totals — never the whole file
 * list rendered at once (`page()`), and a CSV export for the Failed tab's
 * "Download CSV" button.
 */

import { SvelteMap } from 'svelte/reactivity';

export type IngestResultKind =
  'ingested' | 'duplicate' | 'failed' | 'skipped' | 'not_sent';

export interface IngestFileResult {
  identifier: string;
  kind: IngestResultKind;
  error?: string | null;
  /** BA-7: the served stable error code, set whenever `error` is on a
   *  `failed` result. Absent for `not_sent` (never sent — no server
   *  error to carry) and every non-failed kind. */
  error_kind?: string | null;
  image_id?: string | null;
  n_crops?: number | null;
  /** d72cc63: the served secondary-detector failure on an ingested file. */
  secondary_detector_error?: string | null;
}

export interface IngestResults {
  readonly size: number;
  set(id: string, result: IngestFileResult): void;
  get(id: string): IngestFileResult | undefined;
  delete(id: string): void;
  countOf(kind: IngestResultKind): number;
  /** Count of `'failed'` entries matching `errorKind` (`'unknown'` for null/absent). */
  countOfErrorKind(errorKind: string): number;
  failures(): [string, IngestFileResult][];
  /** `not_sent` counts as retryable too — both are what "Retry failed" resends. */
  retryable(): [string, IngestFileResult][];
  /** `errorKind` (undefined = no filter) narrows the `'failed'` page to
   *  entries whose `error_kind` matches (`'unknown'` matches null/absent). */
  page(
    kind: IngestResultKind,
    offset: number,
    limit: number,
    errorKind?: string,
  ): [string, IngestFileResult][];
  /** BA-7: `failed` counts grouped by `error_kind` (`'unknown'` for a
   *  null/absent kind), descending by count — drives the Failed tab's
   *  filter chips. */
  errorKindCounts(): [string, number][];
  toCsv(): string;
  clear(): void;
}

function csvField(v: string | number | null | undefined): string {
  const s = v === null || v === undefined ? '' : String(v);
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

export function createIngestResults(): IngestResults {
  // Plain `$state(new Map())` only makes the *binding* reactive (a full
  // reassignment), not `.set()`/`.delete()` mutation on the same
  // instance — `SvelteMap` (svelte/reactivity) is the built-in that
  // actually tracks per-key mutation, which is exactly how this store
  // is used (`set()`/`delete()` on the same map for the run's lifetime).
  const map = new SvelteMap<string, IngestFileResult>();

  function entriesByKind(kind: IngestResultKind): [string, IngestFileResult][] {
    const out: [string, IngestFileResult][] = [];
    for (const entry of map.entries()) {
      if (entry[1].kind === kind) out.push(entry);
    }
    return out;
  }

  return {
    get size() {
      return map.size;
    },
    set(id, result) {
      map.set(id, result);
    },
    get(id) {
      return map.get(id);
    },
    delete(id) {
      map.delete(id);
    },
    countOf(kind) {
      let n = 0;
      for (const r of map.values()) if (r.kind === kind) n++;
      return n;
    },
    countOfErrorKind(errorKind) {
      let n = 0;
      for (const r of map.values()) {
        if (r.kind === 'failed' && (r.error_kind ?? 'unknown') === errorKind) n++;
      }
      return n;
    },
    failures() {
      return entriesByKind('failed');
    },
    retryable() {
      return [...entriesByKind('failed'), ...entriesByKind('not_sent')];
    },
    page(kind, offset, limit, errorKind) {
      const entries = entriesByKind(kind);
      const filtered =
        errorKind === undefined
          ? entries
          : entries.filter(([, r]) => (r.error_kind ?? 'unknown') === errorKind);
      return filtered.slice(offset, offset + limit);
    },
    errorKindCounts() {
      // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local tally map consumed synchronously within this call, never stored in reactive state
      const counts = new Map<string, number>();
      for (const [, r] of entriesByKind('failed')) {
        const kind = r.error_kind ?? 'unknown';
        counts.set(kind, (counts.get(kind) ?? 0) + 1);
      }
      return [...counts.entries()].sort((a, b) => b[1] - a[1]);
    },
    toCsv() {
      const header = 'identifier,status,error,error_kind,image_id,n_crops';
      const rows = [...map.values()].map((r) =>
        [
          csvField(r.identifier),
          csvField(r.kind),
          csvField(r.error),
          csvField(r.error_kind),
          csvField(r.image_id),
          csvField(r.n_crops),
        ].join(','),
      );
      return [header, ...rows].join('\n');
    },
    clear() {
      map.clear();
    },
  };
}
