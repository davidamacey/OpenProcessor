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
import {
  getAuditQueue,
  getAuditReport,
  startAudit,
  type AuditQueuePage,
} from '$lib/api_labelConfirmation';
import type {
  AuditReport,
  AuditStartRequest,
  AuditStartResponse,
} from '$lib/types_labelConfirmation';

export interface AuditDeps {
  report: (minPerClass?: number | null, signal?: AbortSignal) => Promise<AuditReport>;
  queue: (
    page?: number,
    pageSize?: number,
    batchId?: string | null,
    signal?: AbortSignal,
  ) => Promise<AuditQueuePage>;
  start: (req: AuditStartRequest, signal?: AbortSignal) => Promise<AuditStartResponse>;
}

const DEFAULT_DEPS: AuditDeps = {
  report: getAuditReport,
  queue: getAuditQueue,
  start: startAudit,
};

export const AUDIT_PAGE_SIZE = 30;

export class AuditController {
  loading = $state(true);
  loadError = $state<string | null>(null);
  report = $state<AuditReport | null>(null);
  queue = $state<AuditQueuePage | null>(null);
  queuePage = $state(1);
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
        this.#deps.queue(this.queuePage, AUDIT_PAGE_SIZE, null, signal),
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

  async goToPage(page: number): Promise<void> {
    this.queuePage = page;
    try {
      this.queue = await this.#deps.queue(page, AUDIT_PAGE_SIZE, null);
    } catch (e) {
      this.loadError = detectorErrorLines(e).join(' ');
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
    this.queuePage = 1;
    await this.load();
    return true;
  }
}
