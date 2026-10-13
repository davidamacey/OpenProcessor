/**
 * State of the ingest-sources panel. The served list is the only source of truth:
 * the controller re-reads it after the served `poll_after_s` and renders counts,
 * trigger times and run state as served. Whether a source can run, why not, when
 * it polls next and what a cron means are the backend's answers; this class only
 * holds the form, sends mutations and refreshes.
 */
import { apiErrorText } from '$lib/api';
import {
  createIngestSource,
  deleteIngestSource,
  listIngestSources,
  retryDeadObjects,
  runIngestSource,
  updateIngestSource,
} from '$lib/api_ingestSources';
import type {
  IngestSourceList,
  IngestSourcePatch,
  IngestSourceRequest,
  IngestSourceStatus,
  SourceBackend,
} from '$lib/types_ingestSources';

export interface SourcesDeps {
  listIngestSources: typeof listIngestSources;
  createIngestSource: typeof createIngestSource;
  updateIngestSource: typeof updateIngestSource;
  deleteIngestSource: typeof deleteIngestSource;
  runIngestSource: typeof runIngestSource;
  retryDeadObjects: typeof retryDeadObjects;
}

const DEFAULT_DEPS: SourcesDeps = {
  listIngestSources,
  createIngestSource,
  updateIngestSource,
  deleteIngestSource,
  runIngestSource,
  retryDeadObjects,
};

export interface SourceForm {
  name: string;
  backend: SourceBackend;
  bucket: string;
  prefix: string;
  folder: string;
  extensions: string;
  source: string;
  batchSize: number | null;
  concurrency: number | null;
  watchEnabled: boolean;
  watchInterval: number | null;
  scheduleEnabled: boolean;
  cron: string;
  webhook: boolean;
}

export function blankForm(): SourceForm {
  return {
    name: '',
    backend: 's3',
    bucket: '',
    prefix: '',
    folder: '',
    extensions: '',
    source: 'source',
    batchSize: null,
    concurrency: null,
    watchEnabled: false,
    watchInterval: null,
    scheduleEnabled: false,
    cron: '',
    webhook: false,
  };
}

export function formFromStatus(s: IngestSourceStatus): SourceForm {
  return {
    name: s.name,
    backend: s.backend,
    bucket: s.s3?.bucket ?? '',
    prefix: s.s3?.prefix ?? '',
    folder: s.folder ?? '',
    extensions: s.extensions.join(','),
    source: s.source,
    batchSize: s.batch_size,
    concurrency: s.concurrency,
    watchEnabled: s.triggers.watch?.enabled ?? false,
    watchInterval: s.triggers.watch?.interval_s ?? null,
    scheduleEnabled: s.triggers.schedule?.enabled ?? false,
    cron: s.triggers.schedule?.cron ?? '',
    webhook: s.triggers.webhook_enabled ?? false,
  };
}

function parseExtensions(text: string): string[] | undefined {
  const list = text
    .split(',')
    .map((e) => e.trim())
    .filter((e) => e.length > 0);
  return list.length > 0 ? list : undefined;
}

function triggersOf(f: SourceForm) {
  return {
    watch: {
      enabled: f.watchEnabled,
      ...(f.watchInterval ? { interval_s: f.watchInterval } : {}),
    },
    schedule: { enabled: f.scheduleEnabled, cron: f.cron.trim() },
    webhook_enabled: f.webhook && f.backend === 's3',
  };
}

export function toRequest(f: SourceForm): IngestSourceRequest {
  const body: IngestSourceRequest = {
    name: f.name.trim(),
    backend: f.backend,
    source: f.source.trim() || 'source',
    triggers: triggersOf(f),
  };
  if (f.backend === 's3') body.s3 = { bucket: f.bucket.trim(), prefix: f.prefix.trim() };
  else body.folder = f.folder.trim();
  const ext = parseExtensions(f.extensions);
  if (ext) body.extensions = ext;
  if (f.batchSize) body.batch_size = f.batchSize;
  if (f.concurrency) body.concurrency = f.concurrency;
  return body;
}

export function toPatch(f: SourceForm): IngestSourcePatch {
  const patch: IngestSourcePatch = {
    name: f.name.trim(),
    source: f.source.trim() || 'source',
    triggers: triggersOf(f),
  };
  const ext = parseExtensions(f.extensions);
  if (ext) patch.extensions = ext;
  if (f.batchSize) patch.batch_size = f.batchSize;
  if (f.concurrency) patch.concurrency = f.concurrency;
  return patch;
}

export class IngestSources {
  list = $state<IngestSourceList | null>(null);
  loadError = $state<string | null>(null);
  actionError = $state<string | null>(null);
  notice = $state<string | null>(null);
  busy = $state(false);
  /** `'new'` while the create form is open, a source id while that one is edited. */
  editing = $state<string | 'new' | null>(null);
  form = $state<SourceForm>(blankForm());

  #deps: SourcesDeps;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #stopped = false;

  constructor(deps: Partial<SourcesDeps> = {}) {
    this.#deps = { ...DEFAULT_DEPS, ...deps };
  }

  get sources(): IngestSourceStatus[] {
    return this.list?.items ?? [];
  }

  get canSave(): boolean {
    const f = this.form;
    if (this.busy || f.name.trim().length === 0) return false;
    if (this.editing === 'new') {
      return f.backend === 's3' ? f.bucket.trim().length > 0 : f.folder.trim().length > 0;
    }
    return this.editing !== null;
  }

  async init(): Promise<void> {
    this.#stopped = false;
    await this.refresh();
  }

  stop(): void {
    this.#stopped = true;
    this.#clearTimer();
  }

  #clearTimer(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
  }

  async refresh(): Promise<void> {
    this.#clearTimer();
    let wait = 10;
    try {
      this.list = await this.#deps.listIngestSources();
      this.loadError = null;
      wait = this.list.poll_after_s;
    } catch (e) {
      this.loadError = apiErrorText(e);
    }
    if (!this.#stopped) this.#timer = setTimeout(() => void this.refresh(), wait * 1000);
  }

  openCreate(): void {
    this.actionError = null;
    this.form = blankForm();
    if (this.list && !this.list.s3_configured) this.form.backend = 'folder';
    this.editing = 'new';
  }

  openEdit(source: IngestSourceStatus): void {
    this.actionError = null;
    this.form = formFromStatus(source);
    this.editing = source.source_id;
  }

  closeForm(): void {
    this.editing = null;
    this.actionError = null;
  }

  async save(): Promise<boolean> {
    if (!this.canSave || this.editing === null) return false;
    return this.#act(async () => {
      if (this.editing === 'new')
        await this.#deps.createIngestSource(toRequest(this.form));
      else
        await this.#deps.updateIngestSource(this.editing as string, toPatch(this.form));
      this.editing = null;
    });
  }

  remove(id: string): Promise<boolean> {
    return this.#act(() => this.#deps.deleteIngestSource(id));
  }

  /** Run now. The panel shows the run in the bulk panel's run list; here the
   *  notice names it and the list refresh shows the scan and the counts moving. */
  run(id: string): Promise<boolean> {
    return this.#act(async () => {
      const run = await this.#deps.runIngestSource(id);
      this.notice = `Started ${run.run_id}`;
    });
  }

  retryDead(id: string): Promise<boolean> {
    return this.#act(async () => {
      const result = await this.#deps.retryDeadObjects(id);
      this.notice = `${result.retried} object${result.retried === 1 ? '' : 's'} queued again`;
    });
  }

  async #act(fn: () => Promise<unknown>): Promise<boolean> {
    if (this.busy) return false;
    this.busy = true;
    this.actionError = null;
    this.notice = null;
    try {
      await fn();
      await this.refresh();
      return true;
    } catch (e) {
      this.actionError = apiErrorText(e);
      return false;
    } finally {
      this.busy = false;
    }
  }
}
