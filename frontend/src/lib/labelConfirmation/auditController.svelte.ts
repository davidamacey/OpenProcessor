/**
 * `/audit`: draw a stratified sample of machine-labelled crops for a human
 * to label, then read the detector's and the VLM's measured precision
 * (`POST /audit/start`, `GET /audit/report`), #119. The drawn sample is labelled in the
Review view's `audit` queue.
 *
 * Thin: the sampling, the precision, the Wilson interval, the confusion
 * matrix and the `insufficient_sample` verdict are all served. Nothing here
 * counts, averages or decides whether a class is trustworthy.
 */
import { detectorErrorLines } from '$lib/api_detector';
import { getAuditReport, startAudit } from '$lib/api_labelConfirmation';
import type {
  AuditReport,
  AuditStartRequest,
  AuditStartResponse,
} from '$lib/types_labelConfirmation';

export interface AuditDeps {
  report: (minPerClass?: number | null, signal?: AbortSignal) => Promise<AuditReport>;
  start: (req: AuditStartRequest, signal?: AbortSignal) => Promise<AuditStartResponse>;
}

const DEFAULT_DEPS: AuditDeps = {
  report: getAuditReport,
  start: startAudit,
};

export class AuditController {
  loading = $state(true);
  loadError = $state<string | null>(null);
  report = $state<AuditReport | null>(null);
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
      this.report = await this.#deps.report(this.minPerClass, signal);
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.loadError = detectorErrorLines(e).join(' ');
    } finally {
      this.loading = false;
    }
  }

  /** Draw a sample, then re-read the report. */
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
    await this.load();
    return true;
  }
}
