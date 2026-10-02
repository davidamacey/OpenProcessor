/**
 * One import job's view state (any_domain_plan.md §7.12 item 2, W10.11,
 * W10.12; docs/design/w10-import-reprocess-ui-plan-2026-09-27.md §3).
 *
 * The served job is the only source: the controller re-reads it after
 * the served `poll_after_s` (null = terminal, polling stops), and a
 * `dataset_import.progress|finished` event for this import only wakes an
 * immediate re-read. Status, progress, report counts, errors and the
 * undo report are rendered as served.
 */
import {
  cancelDatasetImport,
  datasetErrorText,
  getDatasetImport,
  getDatasetImportEntries,
  getDatasetImportIssues,
  resumeDatasetImport,
  runServedNextStep,
  undoDatasetImport,
} from '$lib/api';
import { subscribeCurationEvents, type CurationEvent } from '$lib/sse';
import type {
  DatasetImportEntryPage,
  DatasetImportJob,
  DatasetIssuePage,
  DatasetUndoReport,
  NextStep,
} from '$lib/types_import';

/**
 * Which action each served status offers — a reading of W10.11 (resume:
 * "interrupted / failed / cancelled") and W10.12 (undo: not while
 * running). Plan §8 question 8 asks for served `actions` flags instead.
 */
const CANCELLABLE = new Set(['queued', 'running', 'paused_backpressure']);
const RESUMABLE = new Set(['interrupted', 'failed', 'cancelled']);
const UNDOABLE = new Set([
  'completed',
  'completed_with_errors',
  'failed',
  'cancelled',
  'interrupted',
]);

export const PAGE_SIZE = 20;

export interface JobDeps {
  getDatasetImport: typeof getDatasetImport;
  cancelDatasetImport: typeof cancelDatasetImport;
  resumeDatasetImport: typeof resumeDatasetImport;
  undoDatasetImport: typeof undoDatasetImport;
  runServedNextStep: typeof runServedNextStep;
  getDatasetImportIssues: typeof getDatasetImportIssues;
  getDatasetImportEntries: typeof getDatasetImportEntries;
  subscribe: (onEvent: (e: CurationEvent) => void) => { close(): void };
}

const DEFAULT_DEPS: JobDeps = {
  getDatasetImport,
  cancelDatasetImport,
  resumeDatasetImport,
  undoDatasetImport,
  runServedNextStep,
  getDatasetImportIssues,
  getDatasetImportEntries,
  subscribe: (onEvent) => subscribeCurationEvents({ onEvent }),
};

export interface UndoChoices {
  remove_images: boolean;
  deprecate_created_classes: boolean;
}

export class ImportJob {
  readonly importId: string;
  job = $state<DatasetImportJob | null>(null);
  loadError = $state<string | null>(null);
  actionError = $state<string | null>(null);
  busy = $state(false);
  undoReport = $state<DatasetUndoReport | null>(null);

  issues = $state<DatasetIssuePage | null>(null);
  issueCode = $state<string>('');
  issuesPage = $state(1);
  entries = $state<DatasetImportEntryPage | null>(null);
  entryFilters = $state<{ split: string; label_state: string; status: string }>({
    split: '',
    label_state: '',
    status: '',
  });
  entriesPage = $state(1);
  tablesError = $state<string | null>(null);

  #deps: JobDeps;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #sub: { close(): void } | null = null;
  #stopped = false;

  constructor(importId: string, deps: Partial<JobDeps> = {}) {
    this.importId = importId;
    this.#deps = { ...DEFAULT_DEPS, ...deps };
  }

  get status(): string | null {
    return this.job?.status ?? null;
  }

  get canCancel(): boolean {
    return this.status != null && CANCELLABLE.has(this.status);
  }
  get canResume(): boolean {
    return this.status != null && RESUMABLE.has(this.status);
  }
  get canUndo(): boolean {
    return this.status != null && UNDOABLE.has(this.status);
  }

  /** The served label for the current status. */
  statusLabel(fallback: Record<string, string> = {}): string {
    const s = this.status ?? '';
    return this.job?.labels?.['status']?.[s] ?? fallback[s] ?? s;
  }

  start(): void {
    this.#stopped = false;
    void this.load();
    this.#sub = this.#deps.subscribe((e) => this.onEvent(e));
  }

  stop(): void {
    this.#stopped = true;
    this.#clearTimer();
    this.#sub?.close();
    this.#sub = null;
  }

  onEvent(e: CurationEvent): void {
    if (e.type !== 'dataset_import.progress' && e.type !== 'dataset_import.finished')
      return;
    if ((e as { import_id?: unknown }).import_id !== this.importId) return;
    void this.load();
  }

  #clearTimer(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
  }

  #adopt(job: DatasetImportJob): void {
    this.job = job;
    this.#clearTimer();
    if (this.#stopped || job.poll_after_s == null) return;
    this.#timer = setTimeout(
      () => void this.load(),
      Math.max(0, job.poll_after_s) * 1000,
    );
  }

  async load(): Promise<void> {
    try {
      const job = await this.#deps.getDatasetImport(this.importId);
      this.loadError = null;
      this.#adopt(job);
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.loadError = datasetErrorText(e);
      // A transient failure keeps following a running job.
      if (!this.#stopped && this.job?.poll_after_s != null) {
        this.#clearTimer();
        this.#timer = setTimeout(() => void this.load(), this.job.poll_after_s * 1000);
      }
    }
  }

  async #act(fn: () => Promise<DatasetImportJob | null>): Promise<boolean> {
    if (this.busy) return false;
    this.busy = true;
    this.actionError = null;
    try {
      const job = await fn();
      if (job) this.#adopt(job);
      return true;
    } catch (e) {
      this.actionError = datasetErrorText(e);
      return false;
    } finally {
      this.busy = false;
    }
  }

  cancel(): Promise<boolean> {
    return this.#act(() => this.#deps.cancelDatasetImport(this.importId));
  }

  resume(): Promise<boolean> {
    return this.#act(() => this.#deps.resumeDatasetImport(this.importId));
  }

  /** Dry run: the served counts for the confirm dialog. Writes nothing. */
  async undoDryRun(choices: UndoChoices): Promise<DatasetUndoReport | null> {
    if (this.busy) return null;
    this.busy = true;
    this.actionError = null;
    this.undoReport = null;
    try {
      const report = (await this.#deps.undoDatasetImport(this.importId, {
        dry_run: true,
        ...choices,
      })) as DatasetUndoReport;
      this.undoReport = report;
      return report;
    } catch (e) {
      this.actionError = datasetErrorText(e);
      return null;
    } finally {
      this.busy = false;
    }
  }

  undoApply(choices: UndoChoices): Promise<boolean> {
    return this.#act(async () => {
      const job = (await this.#deps.undoDatasetImport(this.importId, {
        dry_run: false,
        ...choices,
      })) as DatasetImportJob;
      this.undoReport = null;
      return job;
    });
  }

  runNextStep(step: NextStep): Promise<boolean> {
    return this.#act(async () => {
      await this.#deps.runServedNextStep(step);
      return null;
    });
  }

  async loadIssues(page = this.issuesPage): Promise<void> {
    try {
      this.issues = await this.#deps.getDatasetImportIssues(this.importId, {
        code: this.issueCode || null,
        page,
        page_size: PAGE_SIZE,
      });
      this.issuesPage = page;
      this.tablesError = null;
    } catch (e) {
      this.tablesError = datasetErrorText(e);
    }
  }

  async loadEntries(page = this.entriesPage): Promise<void> {
    try {
      const f = this.entryFilters;
      this.entries = await this.#deps.getDatasetImportEntries(this.importId, {
        split: f.split || null,
        label_state: f.label_state || null,
        status: f.status || null,
        page,
        page_size: PAGE_SIZE,
      });
      this.entriesPage = page;
      this.tablesError = null;
    } catch (e) {
      this.tablesError = datasetErrorText(e);
    }
  }
}

export function createImportJob(
  importId: string,
  deps: Partial<JobDeps> = {},
): ImportJob {
  return new ImportJob(importId, deps);
}
