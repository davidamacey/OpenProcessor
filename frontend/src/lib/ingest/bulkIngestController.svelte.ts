/**
 * One bulk-ingest panel's state. The served run is the only source: the
 * controller re-reads it after the served `poll_after_s` (null = over, polling
 * stops) and renders status, counts, rates and timings as served. Nothing is
 * computed or decided here; whether a source is acceptable, whether a run is
 * already live and what a start resumes are the backend's answers.
 */
import { apiErrorText } from '$lib/api_errors';
import {
  cancelBulkIngest,
  getBulkIngest,
  getBulkIngestConfig,
  getBulkIngestFailures,
  listBulkIngests,
  startBulkIngest,
} from '$lib/api_bulkIngest';
import type {
  BulkIngestConfig,
  BulkIngestFailurePage,
  BulkIngestRequest,
  BulkIngestRun,
  BulkSourceKind,
} from '$lib/types_bulkIngest';

export const FAILURES_PAGE_SIZE = 20;
const RECENT_RUNS = 5;

export interface BulkDeps {
  getBulkIngestConfig: typeof getBulkIngestConfig;
  startBulkIngest: typeof startBulkIngest;
  listBulkIngests: typeof listBulkIngests;
  getBulkIngest: typeof getBulkIngest;
  getBulkIngestFailures: typeof getBulkIngestFailures;
  cancelBulkIngest: typeof cancelBulkIngest;
}

const DEFAULT_DEPS: BulkDeps = {
  getBulkIngestConfig,
  startBulkIngest,
  listBulkIngests,
  getBulkIngest,
  getBulkIngestFailures,
  cancelBulkIngest,
};

export interface BulkForm {
  kind: BulkSourceKind;
  path: string;
  source: string;
  extensions: string;
  batchSize: number | null;
  concurrency: number | null;
}

export class BulkIngest {
  config = $state<BulkIngestConfig | null>(null);
  configError = $state<string | null>(null);
  run = $state<BulkIngestRun | null>(null);
  recent = $state<BulkIngestRun[]>([]);
  failures = $state<BulkIngestFailurePage | null>(null);
  failuresOffset = $state(0);
  actionError = $state<string | null>(null);
  loadError = $state<string | null>(null);
  busy = $state(false);

  form = $state<BulkForm>({
    kind: 'folder',
    path: '',
    source: 'bulk_ingest',
    extensions: '',
    batchSize: null,
    concurrency: null,
  });

  #deps: BulkDeps;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #stopped = false;

  constructor(deps: Partial<BulkDeps> = {}) {
    this.#deps = { ...DEFAULT_DEPS, ...deps };
  }

  get live(): boolean {
    return this.run?.poll_after_s != null;
  }

  /** The request exactly as the form states it; unset options stay unset so the
   *  backend's served defaults apply. */
  get request(): BulkIngestRequest {
    const f = this.form;
    const body: BulkIngestRequest = {
      [f.kind === 'folder' ? 'folder' : 'manifest']: f.path.trim(),
      source: f.source.trim() || 'bulk_ingest',
    };
    const extensions = f.extensions
      .split(',')
      .map((e) => e.trim())
      .filter((e) => e.length > 0);
    if (extensions.length > 0) body.extensions = extensions;
    if (f.batchSize) body.batch_size = f.batchSize;
    if (f.concurrency) body.concurrency = f.concurrency;
    return body;
  }

  get canStart(): boolean {
    return !this.busy && !this.live && this.form.path.trim().length > 0;
  }

  async init(): Promise<void> {
    this.#stopped = false;
    try {
      this.config = await this.#deps.getBulkIngestConfig();
      this.configError = null;
    } catch (e) {
      this.configError = apiErrorText(e);
      return;
    }
    await this.refreshRecent();
    // Pick up a run started from another tab or the command line.
    const newest = this.recent[0];
    if (newest && !this.run) this.#adopt(newest);
  }

  stop(): void {
    this.#stopped = true;
    this.#clearTimer();
  }

  async refreshRecent(): Promise<void> {
    try {
      this.recent = (await this.#deps.listBulkIngests({ page_size: RECENT_RUNS })).items;
      this.loadError = null;
    } catch (e) {
      this.loadError = apiErrorText(e);
    }
  }

  #clearTimer(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
  }

  #adopt(run: BulkIngestRun): void {
    this.run = run;
    this.#clearTimer();
    if (run.poll_after_s == null) {
      void this.#finished(run);
      return;
    }
    if (this.#stopped) return;
    this.#timer = setTimeout(
      () => void this.poll(),
      Math.max(0, run.poll_after_s) * 1000,
    );
  }

  /** Show an earlier run (and follow it when it is live). */
  select(run: BulkIngestRun): void {
    this.failures = null;
    this.#adopt(run);
  }

  async #finished(run: BulkIngestRun): Promise<void> {
    await this.refreshRecent();
    if (run.progress.failed > 0) await this.loadFailures(0);
    else this.failures = null;
  }

  async poll(): Promise<void> {
    const current = this.run;
    if (!current) return;
    try {
      const next = await this.#deps.getBulkIngest(current.run_id);
      this.loadError = null;
      this.#adopt(next);
    } catch (e) {
      this.loadError = apiErrorText(e);
      // A transient failure keeps following a live run.
      if (!this.#stopped && current.poll_after_s != null) {
        this.#clearTimer();
        this.#timer = setTimeout(() => void this.poll(), current.poll_after_s * 1000);
      }
    }
  }

  async start(request: BulkIngestRequest = this.request): Promise<boolean> {
    if (this.busy) return false;
    this.busy = true;
    this.actionError = null;
    this.failures = null;
    try {
      this.#adopt(await this.#deps.startBulkIngest(request));
      return true;
    } catch (e) {
      this.actionError = apiErrorText(e);
      return false;
    } finally {
      this.busy = false;
    }
  }

  /** Start the same source again: the backend continues the run it left off. */
  resume(run: BulkIngestRun): Promise<boolean> {
    // A registered source runs with its own options; a path source repeats the run's.
    const body: BulkIngestRequest =
      run.source.kind === 'queue'
        ? { source_id: run.source.path }
        : {
            [run.source.kind === 'folder' ? 'folder' : 'manifest']: run.source.path,
            source: run.options.source_tag,
            extensions: run.source.extensions,
            batch_size: run.options.batch_size,
            concurrency: run.options.concurrency,
          };
    return this.start(body);
  }

  async cancel(): Promise<boolean> {
    const current = this.run;
    if (!current || this.busy) return false;
    this.busy = true;
    this.actionError = null;
    try {
      this.#adopt(await this.#deps.cancelBulkIngest(current.run_id));
      return true;
    } catch (e) {
      this.actionError = apiErrorText(e);
      return false;
    } finally {
      this.busy = false;
    }
  }

  async loadFailures(offset: number): Promise<void> {
    const current = this.run;
    if (!current) return;
    try {
      this.failures = await this.#deps.getBulkIngestFailures(current.run_id, {
        offset,
        limit: FAILURES_PAGE_SIZE,
      });
      this.failuresOffset = offset;
    } catch (e) {
      this.loadError = apiErrorText(e);
    }
  }
}
