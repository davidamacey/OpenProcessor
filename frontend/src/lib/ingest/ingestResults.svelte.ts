/**
 * Compact per-file result storage for an ingest run
 * (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §A.2). A
 * `Map<id, IngestFileResult>` plus running totals — never the whole file
 * list rendered at once (`page()`), and a CSV export for the Failed tab's
 * "Download CSV" button.
 */

import { SvelteMap } from 'svelte/reactivity';

export type IngestResultKind =
  | 'ingested'
  | 'duplicate'
  | 'failed'
  | 'skipped'
  | 'not_sent';

export interface IngestFileResult {
  identifier: string;
  kind: IngestResultKind;
  error?: string | null;
  image_id?: string | null;
  n_crops?: number | null;
}

export interface IngestResults {
  readonly size: number;
  set(id: string, result: IngestFileResult): void;
  get(id: string): IngestFileResult | undefined;
  delete(id: string): void;
  countOf(kind: IngestResultKind): number;
  failures(): [string, IngestFileResult][];
  /** `not_sent` counts as retryable too — both are what "Retry failed" resends. */
  retryable(): [string, IngestFileResult][];
  page(
    kind: IngestResultKind,
    offset: number,
    limit: number,
  ): [string, IngestFileResult][];
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
    failures() {
      return entriesByKind('failed');
    },
    retryable() {
      return [...entriesByKind('failed'), ...entriesByKind('not_sent')];
    },
    page(kind, offset, limit) {
      return entriesByKind(kind).slice(offset, offset + limit);
    },
    toCsv() {
      const header = 'identifier,status,error,image_id,n_crops';
      const rows = [...map.values()].map((r) =>
        [
          csvField(r.identifier),
          csvField(r.kind),
          csvField(r.error),
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
