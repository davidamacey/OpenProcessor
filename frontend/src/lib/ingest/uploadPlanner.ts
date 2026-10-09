/**
 * Pure chunk planner for an ingest run
 * (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §A.2/§A.4).
 * Packs a flat file list into request-sized chunks respecting both an
 * image-count cap and a byte cap, greedily and in input order.
 */

import type { IngestFile } from './fileSource';

export interface ChunkPlanOptions {
  maxImages: number;
  maxBytes: number;
}

/** A file too large to ever fit `maxBytes` on its own gets isolated into
 *  its own single-file chunk and flagged — callers must never send it. */
export interface OversizeFile extends IngestFile {
  oversize: true;
}

export function isOversize(f: IngestFile | OversizeFile): f is OversizeFile {
  return (f as OversizeFile).oversize === true;
}

/**
 * Order is preserved: chunk N's files all precede chunk N+1's files in
 * the input order, and within a chunk order is preserved too.
 */
export function planChunks(
  files: IngestFile[],
  opts: ChunkPlanOptions,
): (IngestFile | OversizeFile)[][] {
  const chunks: (IngestFile | OversizeFile)[][] = [];
  let current: IngestFile[] = [];
  let currentBytes = 0;

  const flush = () => {
    if (current.length > 0) {
      chunks.push(current);
      current = [];
      currentBytes = 0;
    }
  };

  for (const f of files) {
    if (f.size > opts.maxBytes) {
      flush();
      chunks.push([{ ...f, oversize: true }]);
      continue;
    }
    const wouldExceedBytes = currentBytes + f.size > opts.maxBytes;
    const wouldExceedCount = current.length + 1 > opts.maxImages;
    if (current.length > 0 && (wouldExceedBytes || wouldExceedCount)) {
      flush();
    }
    current.push(f);
    currentBytes += f.size;
  }
  flush();
  return chunks;
}

/** Chunks a flat id list for `path_lookup`'s `maxItems` cap. */
export function chunkForLookup(ids: string[], max: number): string[][] {
  if (max <= 0) return ids.length ? [ids] : [];
  const out: string[][] = [];
  for (let i = 0; i < ids.length; i += max) {
    out.push(ids.slice(i, i + max));
  }
  return out;
}
