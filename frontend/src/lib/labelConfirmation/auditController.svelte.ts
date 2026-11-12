/**
 * `/audit`: draw a stratified sample of machine-labelled crops for a human
 * to label, then read the detector's and the VLM's measured precision
 * (`POST /audit/start`, `GET /audit/queue`, `GET /audit/report`), #119.
 *
 * Thin: the sampling, the precision, the Wilson interval, the confusion
 * matrix and the `insufficient_sample` verdict are all served. Nothing here
 * counts, averages or decides whether a class is trustworthy.
 */
import { detectorErrorLines } from '$lib/api_detector';
import { getAuditQueue, getAuditReport, startAudit } from '$lib/api_labelConfirmation';
import type { Crop } from '$lib/types';
import type { CursorPage } from '$lib/types_cursor';
import type {
  AuditReport,
  AuditStartRequest,
  AuditStartResponse,
} from '$lib/types_labelConfirmation';

export interface AuditDeps {
  report: (minPerClass?: number | null, signal?: AbortSignal) => Promise<AuditReport>;
  queue: (
    cursor?: string | null,
    limit?: number,
    batchId?: string | null,
    signal?: AbortSignal,
  ) => Promise<CursorPage<Crop>>;
  start: (req: AuditStartRequest, signal?: AbortSignal) => Promise<AuditStartResponse>;
}

const DEFAULT_DEPS: AuditDeps = {
  report: getAuditReport,
  queue: getAuditQueue,
  start: startAudit,
};

const AUDIT_PAGE_SIZE = 30;

export class AuditController {
  loading = $state(true);
  loadError = $state<string | null>(null);
  report = $state<AuditReport | null>(null);
  queue = $state<CursorPage<Crop> | null>(null);
  /** The cursors of the pages before the one shown, nearest last. */
  #before = $state<(string | null)[]>([]);
  #cursor = $state<string | null>(null);
  hasPrevious = $derived(this.#before.length > 0);
  /** Operator inputs; an empty one is not sent, so the server default applies. */
  sampleSize = $state<number | null>(null);
  minPerClass = $state<number | null>(null);

  starting = $state(false);
  startLines = $state<string[]>([]);
  /** The last draw's served summary (strata, how many were sampled). */
  started = $state<AuditStartResponse | null>(null);

  #deps: AuditDeps;

  constructor(deps: AuditDeps = DEFAULT_DEPS) {
    this.#deps = deps;
  }

  async load(signal?: AbortSignal): Promise<void> {
    this.loading = true;
    this.loadError = null;
    try {
      const [report, queue] = await Promise.all([
        this.#deps.report(this.minPerClass, signal),
        this.#deps.queue(this.#cursor, AUDIT_PAGE_SIZE, null, signal),
      ]);
      this.report = report;
      this.queue = queue;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.loadError = detectorErrorLines(e).join(' ');
    } finally {
      this.loading = false;
    }
  }

  async next(): Promise<void> {
    const target = this.queue?.next_cursor;
    if (!target) return;
    const shown = this.#cursor;
    if (await this.#show(target)) {
      this.#before = [...this.#before, shown];
      this.#cursor = target;
    }
  }

  async previous(): Promise<void> {
    const cursor = this.#before.at(-1);
    if (cursor === undefined) return;
    if (await this.#show(cursor)) {
      this.#before = this.#before.slice(0, -1);
      this.#cursor = cursor;
    }
  }

  async #show(cursor: string | null): Promise<boolean> {
    try {
      this.queue = await this.#deps.queue(cursor, AUDIT_PAGE_SIZE, null);
      return true;
    } catch (e) {
      this.loadError = detectorErrorLines(e).join(' ');
      return false;
    }
  }

  /** Draw a sample, then re-read the report and the first queue page. */
  async start(): Promise<boolean> {
    this.starting = true;
    this.startLines = [];
    try {
      this.started = await this.#deps.start({
        min_per_class: this.minPerClass ?? undefined,
        sample_size: this.sampleSize ?? undefined,
      });
    } catch (e) {
      this.startLines = detectorErrorLines(e);
      return false;
    } finally {
      this.starting = false;
    }
    this.#before = [];
    this.#cursor = null;
    await this.load();
    return true;
  }
}
