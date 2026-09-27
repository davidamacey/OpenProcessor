/**
 * `/ingest`'s run state machine — the only place upload concurrency
 * lives. Factory-function convention, matching `clusterController`/
 * `reviewController` (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md
 * §A.4).
 *
 * States: `idle → enumerating → prefiltering → uploading ⇄ paused →
 * (done | cancelled | error)`. `enumerating` is owned by the caller
 * (file-picker/drop collection happens before `start()`); this
 * controller's own state starts at `prefiltering`.
 *
 * **Plan deviation (recorded per CLAUDE.md's "trust the code" rule):**
 * the plan's §A.4 table describes a nginx-style 413 as carrying an HTML
 * body with `ApiError.detail == null`. `api.ts`'s `errorDetail()` does
 * not do that — it returns the raw HTML *text* as `detail` when the body
 * isn't JSON (any string body sets `raw = body` unconditionally), so a
 * nginx 413 has a non-null but nonsensical `detail`, indistinguishable
 * from a backend 413's `detail` by string content alone. The actual
 * distinguishing signal, verified by reading `apiFetch`, is `e.body`'s
 * *type*: a backend 413 parses as JSON (`typeof e.body === 'object'`); an
 * nginx 413 fails `res.json()` and falls back to `res.text()`
 * (`typeof e.body === 'string'`). This controller keys off `e.body`'s
 * type instead of `e.detail === null`.
 */

import { ApiError } from '$lib/api';
import type {
  BatchIngestResponse,
  IngestPathLookupResponse,
  IngestUploadRequest,
} from '$lib/types';
import type { IngestFile } from './fileSource';
import {
  chunkForLookup,
  isOversize,
  planChunks,
  type OversizeFile,
} from './uploadPlanner';
import type { ResolvedIngestConfig } from './ingestConfig';
import { createIngestResults, type IngestResults } from './ingestResults.svelte';
import { toastStore } from '$stores/toast.svelte';

export type IngestRunState =
  'idle' | 'prefiltering' | 'uploading' | 'paused' | 'done' | 'cancelled' | 'error';

export interface IngestTotals {
  queued: number;
  skipped_known: number;
  uploaded_bytes: number;
  successful: number;
  duplicates: number;
  failed: number;
  crops_indexed: number;
  /** d72cc63: sum of the served `summary.secondary_detector_failures`. */
  secondary_detector_failures: number;
}

export interface IngestRunStartOpts {
  source: string;
  identifierPrefix: string;
  skipLookup: boolean;
}

export interface IngestRunDeps {
  lookup: (ids: string[], signal?: AbortSignal) => Promise<IngestPathLookupResponse>;
  upload: (
    req: IngestUploadRequest,
    signal?: AbortSignal,
  ) => Promise<BatchIngestResponse>;
  config: ResolvedIngestConfig;
  /** Default 2 — see §A.4: the browser allows 6 connections/origin and
   *  the page still has to poll status/drain/health. */
  concurrency?: number;
  onChunkDone?: () => void;
}

export interface IngestRun {
  readonly state: IngestRunState;
  readonly totals: IngestTotals;
  readonly results: IngestResults;
  /** Set while `state === 'paused'` from a served 503, else `null`. */
  readonly pauseReason: string | null;
  /** Set once `state === 'error'` (the nginx-413 stop case). */
  readonly errorReason: string | null;
  start(files: IngestFile[], opts: IngestRunStartOpts): Promise<void>;
  pause(): void;
  resume(): void;
  cancel(): void;
  retryFailed(): Promise<void>;
}

type Chunk = (IngestFile | OversizeFile)[];

function emptyTotals(): IngestTotals {
  return {
    queued: 0,
    skipped_known: 0,
    uploaded_bytes: 0,
    successful: 0,
    duplicates: 0,
    failed: 0,
    crops_indexed: 0,
    secondary_detector_failures: 0,
  };
}

export function createIngestRun(deps: IngestRunDeps): IngestRun {
  let state = $state<IngestRunState>('idle');
  const totals = $state<IngestTotals>(emptyTotals());
  const results = createIngestResults();
  let pauseReason = $state<string | null>(null);
  let errorReason = $state<string | null>(null);

  const concurrency = deps.concurrency ?? 2;

  let runOpts: IngestRunStartOpts | null = null;
  let filesById = new Map<string, IngestFile>();
  let queue: Chunk[] = [];
  let cursor = 0;
  let inFlight = 0;
  let pauseGate: Promise<void> | null = null;
  let releasePauseGate: (() => void) | null = null;
  let cancelRequested = false;
  let abortController = new AbortController();
  const halvedOnce = new WeakSet<object>();

  function identifierFor(f: IngestFile): string {
    return `${runOpts!.identifierPrefix}${f.relPath}`;
  }

  function waitIfPaused(): Promise<void> {
    return pauseGate ?? Promise.resolve();
  }

  function armPauseGate(): void {
    if (pauseGate) return;
    pauseGate = new Promise((resolve) => {
      releasePauseGate = resolve;
    });
  }

  function releasePause(): void {
    releasePauseGate?.();
    pauseGate = null;
    releasePauseGate = null;
  }

  async function runPrefilter(files: IngestFile[]): Promise<IngestFile[]> {
    if (runOpts!.skipLookup) return files;
    const idToFile = new Map(files.map((f) => [identifierFor(f), f]));
    const idChunks = chunkForLookup([...idToFile.keys()], deps.config.pathLookupMax);
    const known = new Set<string>();
    for (const chunk of idChunks) {
      try {
        const res = await deps.lookup(chunk, abortController.signal);
        for (const id of Object.keys(res.known_paths)) known.add(id);
      } catch {
        toastStore.warn('pre-check unavailable — relying on server content dedup');
        break;
      }
    }
    const remaining: IngestFile[] = [];
    for (const f of files) {
      const id = identifierFor(f);
      if (known.has(id)) {
        results.set(f.id, { identifier: id, kind: 'skipped', error: 'already indexed' });
        totals.skipped_known++;
      } else {
        remaining.push(f);
      }
    }
    return remaining;
  }

  function planUploadChunks(files: IngestFile[]): Chunk[] {
    const planned = planChunks(files, {
      maxImages: deps.config.uploadMaxImages,
      maxBytes: deps.config.uploadMaxBytes,
    });
    const chunks: Chunk[] = [];
    for (const chunk of planned) {
      if (chunk.length === 1 && isOversize(chunk[0]!)) {
        const f = chunk[0]!;
        results.set(f.id, {
          identifier: identifierFor(f),
          kind: 'failed',
          error: "larger than this deployment's upload limit",
        });
        totals.failed++;
        continue;
      }
      chunks.push(chunk);
    }
    return chunks;
  }

  function applyResponse(chunk: Chunk, res: BatchIngestResponse): void {
    // For an upload result `image_path` is the server-persisted path, not
    // the client identifier — the identifier this controller sent
    // (`image_paths` form field) comes back as `source_identifier`.
    const byIdentifier = new Map(
      res.results
        .filter((r) => r.source_identifier != null)
        .map((r) => [r.source_identifier, r]),
    );
    // Defensive fallback for an in-batch byte-identical duplicate: today's
    // backend can return the *second* copy of a duplicate pair with
    // `source_identifier: null` (a server-side fix is in progress), so
    // that row has no key matching any file's own identifier and used to
    // render as "failed — no result returned" even though the backend
    // actually answered for it. When a result can't be matched by
    // identifier AND the response has exactly as many rows as files sent
    // in this chunk, fall back to matching that file by request order
    // instead — never used when the lengths differ, since that's the
    // signal a result genuinely didn't come back at all.
    const canFallBackToRequestOrder = res.results.length === chunk.length;
    totals.secondary_detector_failures += res.summary?.secondary_detector_failures ?? 0;
    for (let i = 0; i < chunk.length; i++) {
      const f = chunk[i]!;
      const id = identifierFor(f);
      let r = byIdentifier.get(id);
      if (!r && canFallBackToRequestOrder) {
        const positional = res.results[i];
        if (positional && positional.source_identifier == null) {
          r = positional;
        }
      }
      totals.uploaded_bytes += f.size;
      if (!r) {
        results.set(f.id, {
          identifier: id,
          kind: 'failed',
          error: 'no result returned',
        });
        totals.failed++;
        continue;
      }
      if (r.status === 'success') {
        results.set(f.id, {
          identifier: id,
          kind: 'ingested',
          image_id: r.image_id,
          n_crops: r.n_crops,
          secondary_detector_error: r.secondary_detector_error ?? null,
        });
        totals.successful++;
        totals.crops_indexed += r.n_crops;
      } else if (r.status === 'duplicate') {
        results.set(f.id, { identifier: id, kind: 'duplicate', image_id: r.image_id });
        totals.duplicates++;
      } else {
        results.set(f.id, {
          identifier: id,
          kind: 'failed',
          error: r.error,
          error_kind: r.error_kind,
        });
        totals.failed++;
      }
    }
  }

  function markChunk(
    chunk: Chunk,
    kind: 'failed' | 'not_sent',
    error?: string | null,
    errorKind?: string | null,
  ): void {
    for (const f of chunk) {
      const id = identifierFor(f);
      if (kind === 'failed') totals.failed++;
      results.set(f.id, {
        identifier: id,
        kind,
        error: error ?? null,
        error_kind: errorKind,
      });
    }
  }

  async function runChunk(chunk: Chunk): Promise<void> {
    const req: IngestUploadRequest = {
      files: chunk.map((f) => f.file),
      identifiers: chunk.map(identifierFor),
      source: runOpts!.source,
    };
    try {
      const res = await deps.upload(req, abortController.signal);
      applyResponse(chunk, res);
    } catch (e) {
      if (e instanceof ApiError && e.status === 413) {
        const isBackendStyle = e.body !== null && typeof e.body === 'object';
        if (isBackendStyle) {
          if (!halvedOnce.has(chunk) && chunk.length > 1) {
            halvedOnce.add(chunk);
            const mid = Math.ceil(chunk.length / 2);
            const a = chunk.slice(0, mid);
            const b = chunk.slice(mid);
            halvedOnce.add(a);
            halvedOnce.add(b);
            queue.splice(cursor, 0, a, b);
          } else {
            markChunk(chunk, 'failed', e.detail ?? 'request too large', 'too_large');
          }
        } else {
          // nginx-style 413 — HTML body, not JSON. Stop the whole run;
          // do not retry.
          state = 'error';
          errorReason =
            `Upload rejected by the Cropwright proxy — request larger than ` +
            `CROPWRIGHT_INGEST_MAX_REQUEST_MB`;
          cancelRequested = true;
          toastStore.error(errorReason);
          markChunk(chunk, 'not_sent');
        }
      } else if (e instanceof ApiError && e.status === 422) {
        markChunk(chunk, 'failed', e.detail ?? 'validation failed');
      } else if (e instanceof ApiError && e.status === 503) {
        // Requeue this exact chunk so Resume retries it.
        queue.splice(cursor, 0, chunk);
        if (state !== 'error') {
          state = 'paused';
          pauseReason = e.detail ?? 'backend unavailable';
          armPauseGate();
          toastStore.warn(`Upload paused — ${pauseReason}`);
        }
      } else {
        // Network error (retries already exhausted by apiFetch) or an
        // abort. Both are safe to leave as "not sent": a rerun resumes
        // via server-side content dedup, per §A.4.
        markChunk(chunk, 'not_sent');
      }
    }
  }

  async function worker(): Promise<void> {
    for (;;) {
      if (cancelRequested) return;
      await waitIfPaused();
      if (cancelRequested) return;
      if (cursor >= queue.length) return;
      const chunk = queue[cursor]!;
      cursor++;
      inFlight++;
      await runChunk(chunk);
      inFlight--;
      deps.onChunkDone?.();
      if (cancelRequested) return;
    }
  }

  async function dispatch(): Promise<void> {
    state = 'uploading';
    const workers = Array.from({ length: concurrency }, () => worker());
    await Promise.all(workers);

    // A worker only returns from its loop once the queue is fully
    // drained or `cancelRequested` is set — never while merely paused
    // (a paused worker blocks on `waitIfPaused()` until `resume()` or
    // `cancel()` releases the gate), so there is no "still paused"
    // branch to handle here.
    // Read through a function so TS doesn't narrow `state` to the
    // literal it was last assigned in this same function scope — a
    // worker's runChunk() can reassign it (to 'error' on an nginx-style
    // 413) from a different call frame in between.
    const currentState = (): IngestRunState => state;
    if (cancelRequested) {
      if (currentState() !== 'error') state = 'cancelled';
      toastStore.info('Upload run cancelled');
    } else if (currentState() !== 'error') {
      state = 'done';
      toastStore.success('Upload run complete');
    }
  }

  return {
    get state() {
      return state;
    },
    get totals() {
      return totals;
    },
    get results() {
      return results;
    },
    get pauseReason() {
      return pauseReason;
    },
    get errorReason() {
      return errorReason;
    },

    async start(files, opts) {
      runOpts = opts;
      cancelRequested = false;
      pauseReason = null;
      errorReason = null;
      Object.assign(totals, emptyTotals());
      results.clear();
      filesById = new Map(files.map((f) => [f.id, f]));
      abortController = new AbortController();

      state = 'prefiltering';
      const remaining = await runPrefilter(files);
      if (cancelRequested) {
        state = 'cancelled';
        return;
      }
      queue = planUploadChunks(remaining);
      cursor = 0;
      totals.queued = queue.reduce((n, c) => n + c.length, 0);
      if (totals.queued === 0 && totals.skipped_known > 0 && totals.failed === 0) {
        // F-59: a re-upload of already-indexed files used to toast
        // "Upload started (0 images queued)" and then show nothing.
        toastStore.info(
          `Nothing to upload: ${totals.skipped_known} already indexed (see the Skipped tab)`,
        );
      } else {
        toastStore.info(`Upload started (${totals.queued} images queued)`);
      }
      await dispatch();
    },

    pause() {
      if (state !== 'uploading') return;
      state = 'paused';
      armPauseGate();
    },

    resume() {
      if (state !== 'paused') return;
      pauseReason = null;
      state = 'uploading';
      // The workers spawned by the original dispatch() are still alive,
      // blocked inside `waitIfPaused()` — releasing the gate wakes them
      // in place. No re-dispatch needed (see dispatch()'s own comment).
      releasePause();
    },

    cancel() {
      cancelRequested = true;
      abortController.abort();
      releasePause();
      for (let i = cursor; i < queue.length; i++) {
        markChunk(queue[i]!, 'not_sent');
      }
      queue = queue.slice(0, cursor);
      if (inFlight === 0) state = 'cancelled';
    },

    async retryFailed() {
      const retryable = results.retryable();
      if (retryable.length === 0) return;
      const files: IngestFile[] = [];
      for (const [id] of retryable) {
        const f = filesById.get(id);
        if (f) files.push(f);
      }
      for (const [id, r] of retryable) {
        if (r.kind === 'failed') totals.failed--;
        results.delete(id);
      }
      cancelRequested = false;
      abortController = new AbortController();
      queue = planUploadChunks(files);
      cursor = 0;
      totals.queued += queue.reduce((n, c) => n + c.length, 0);
      await dispatch();
    },
  };
}
