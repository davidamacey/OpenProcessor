/**
 * The dataset-import wizard's state (any_domain_plan.md §7.12 item 1;
 * docs/design/w10-import-reprocess-ui-plan-2026-09-27.md §2).
 *
 * Holds only what the operator chose (source, explicit mapping rows,
 * touched options) and what the server answered (the preview, the start
 * outcome). Every count, suggestion, resolved target, issue and verdict
 * is the served preview's; the client never compares a dataset class id
 * with a registry id, and never decides which suggestions
 * `accept_suggestions` covers — the preview's `resolved` says.
 */
import {
  datasetErrorDetail,
  apiErrorText,
  previewDataset,
  resumeDatasetImport,
  startDatasetImport,
  uploadDatasetArchive,
} from '$lib/api';
import type {
  ClassMappingEntry,
  DatasetClassRow,
  DatasetImportJob,
  DatasetImportOptions,
  DatasetImportRequest,
  DatasetIssue,
  DatasetPreview,
  DatasetPreviewRequest,
} from '$lib/types_import';

export const PREVIEW_DEBOUNCE_MS = 400;

/** One explicit mapping choice. `action: ''` = not chosen (row omitted). */
export interface MappingChoice {
  action: string;
  class_id: number | null;
  new_class_name: string;
}

export interface StartRefusal {
  code: string | null;
  message: string;
  importId: string | null;
  issues: DatasetIssue[];
}

export interface WizardDeps {
  previewDataset: typeof previewDataset;
  startDatasetImport: typeof startDatasetImport;
  resumeDatasetImport: typeof resumeDatasetImport;
  uploadDatasetArchive: typeof uploadDatasetArchive;
}

const DEFAULT_DEPS: WizardDeps = {
  previewDataset,
  startDatasetImport,
  resumeDatasetImport,
  uploadDatasetArchive,
};

export class ImportWizard {
  sourcePath = $state('');
  format = $state('auto');
  acceptSuggestions = $state(false);
  /** Explicit mapping choices by dataset class name. */
  choices = $state<Record<string, MappingChoice>>({});
  /** Options the operator set; everything else is the server's default. */
  options = $state<DatasetImportOptions>({});

  preview = $state<DatasetPreview | null>(null);
  previewError = $state<string | null>(null);
  previewing = $state(false);
  /** Inputs changed since the preview on screen was requested. */
  stale = $state(false);

  starting = $state(false);
  refusal = $state<StartRefusal | null>(null);
  /** Dataset classes the server's 422 `class_mapping_incomplete` named. */
  unmappedFromServer = $state<string[]>([]);
  reusedJob = $state<DatasetImportJob | null>(null);

  uploading = $state(false);
  uploadError = $state<string | null>(null);

  #deps: WizardDeps;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #abort: AbortController | null = null;
  #seq = 0;

  constructor(deps: Partial<WizardDeps> = {}) {
    this.#deps = { ...DEFAULT_DEPS, ...deps };
  }

  // -- inputs -------------------------------------------------------------

  setSource(path: string): void {
    this.sourcePath = path;
    this.#changed();
  }

  setFormat(format: string): void {
    this.format = format;
    this.#changed();
  }

  setAcceptSuggestions(v: boolean): void {
    this.acceptSuggestions = v;
    this.#changed();
  }

  choiceFor(datasetClass: string): MappingChoice {
    return (
      this.choices[datasetClass] ?? { action: '', class_id: null, new_class_name: '' }
    );
  }

  setChoice(datasetClass: string, patch: Partial<MappingChoice>): void {
    this.choices = {
      ...this.choices,
      [datasetClass]: { ...this.choiceFor(datasetClass), ...patch },
    };
    this.#changed();
  }

  /** Copy the served suggestion into an explicit choice. */
  useSuggestion(row: DatasetClassRow): void {
    const s = row.suggestion;
    if (!s) return;
    this.setChoice(row.dataset_class, {
      action: s.action,
      class_id: s.class_id,
      new_class_name: s.action === 'create' ? (s.class_name ?? row.dataset_class) : '',
    });
  }

  setOption<K extends keyof DatasetImportOptions>(
    key: K,
    value: DatasetImportOptions[K] | undefined,
  ): void {
    const next = { ...this.options };
    if (value === undefined) delete next[key];
    else next[key] = value;
    this.options = next;
    this.#changed();
  }

  /**
   * Pre-fill the explicit choices from a previous import's served
   * `mapping` ("use the mapping from import X"). Names only; rows for
   * classes this dataset doesn't have are ignored.
   */
  prefillFromJob(job: DatasetImportJob): void {
    const known = (this.preview?.classes ?? []).map((c) => c.dataset_class);
    const next = { ...this.choices };
    for (const t of job.mapping) {
      if (!t.dataset_class || !known.includes(t.dataset_class)) continue;
      if (t.kind === 'item' && t.class_id != null) {
        next[t.dataset_class] = {
          action: 'map',
          class_id: t.class_id,
          new_class_name: '',
        };
      } else if (t.kind === 'region' || t.kind === 'skip') {
        next[t.dataset_class] = { action: t.kind, class_id: null, new_class_name: '' };
      }
    }
    this.choices = next;
    this.#changed();
  }

  // -- request ------------------------------------------------------------

  mappingEntries(): ClassMappingEntry[] {
    const out: ClassMappingEntry[] = [];
    for (const [dataset_class, c] of Object.entries(this.choices)) {
      if (!c.action) continue;
      const entry: ClassMappingEntry = { dataset_class, action: c.action };
      if (c.action === 'map') entry.class_id = c.class_id;
      if (c.action === 'create') entry.new_class_name = c.new_class_name;
      out.push(entry);
    }
    return out;
  }

  requestBody(): DatasetPreviewRequest {
    const options = { ...this.options };
    // "Start anyway" only exists while the served preview allows it.
    if (!this.preview?.force_allowed) delete options.force;
    return {
      source: { path: this.sourcePath.trim(), format: this.format },
      mapping: this.mappingEntries(),
      accept_suggestions: this.acceptSuggestions,
      options,
    };
  }

  // -- preview ------------------------------------------------------------

  #changed(): void {
    this.stale = true;
    this.refusal = null;
    this.reusedJob = null;
    this.schedulePreview();
  }

  schedulePreview(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = setTimeout(() => {
      this.#timer = null;
      void this.runPreview();
    }, PREVIEW_DEBOUNCE_MS);
  }

  async runPreview(): Promise<void> {
    if (this.#timer) {
      clearTimeout(this.#timer);
      this.#timer = null;
    }
    if (!this.sourcePath.trim()) {
      this.preview = null;
      this.previewError = null;
      this.stale = false;
      return;
    }
    this.#abort?.abort();
    const ctrl = new AbortController();
    this.#abort = ctrl;
    const seq = ++this.#seq;
    this.previewing = true;
    try {
      const p = await this.#deps.previewDataset(this.requestBody(), ctrl.signal);
      if (seq !== this.#seq) return;
      this.preview = p;
      this.previewError = null;
      this.stale = false;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError' || seq !== this.#seq) return;
      this.previewError = apiErrorText(e);
      this.preview = null;
      this.stale = false;
    } finally {
      if (seq === this.#seq) this.previewing = false;
    }
  }

  /** Rows the served preview says the request doesn't cover yet. */
  get unmappedRows(): DatasetClassRow[] {
    const fromServer = this.unmappedFromServer;
    return (this.preview?.classes ?? []).filter(
      (r) => (r.boxes > 0 && r.resolved == null) || fromServer.includes(r.dataset_class),
    );
  }

  get canStart(): boolean {
    const p = this.preview;
    if (!p || this.previewing || this.stale || this.starting) return false;
    if (p.blocking && !(p.force_allowed && this.options.force === true)) return false;
    return this.unmappedRows.length === 0;
  }

  // -- start / resume -----------------------------------------------------

  /**
   * Returns the new job to follow, or `null` when the server refused, or
   * answered with an already-completed import (`reusedJob`).
   */
  async start(): Promise<DatasetImportJob | null> {
    const p = this.preview;
    if (!p || this.starting) return null;
    this.starting = true;
    this.refusal = null;
    this.reusedJob = null;
    this.unmappedFromServer = [];
    const body: DatasetImportRequest = {
      ...this.requestBody(),
      expected_import_key: p.import_key,
    };
    try {
      const job = await this.#deps.startDatasetImport(body);
      if (job.reused) {
        this.reusedJob = job;
        return null;
      }
      return job;
    } catch (e) {
      const d = datasetErrorDetail(e);
      this.refusal = {
        code: d?.error ?? null,
        message: apiErrorText(e),
        importId: d?.import_id ?? null,
        issues: d?.issues ?? [],
      };
      if (d?.error === 'class_mapping_incomplete')
        this.unmappedFromServer = d.unmapped ?? [];
      if (d?.error === 'dataset_changed') void this.runPreview();
      return null;
    } finally {
      this.starting = false;
    }
  }

  /** Resume the import a 409 `import_resumable` named. */
  async resume(importId: string): Promise<DatasetImportJob | null> {
    try {
      return await this.#deps.resumeDatasetImport(importId);
    } catch (e) {
      const d = datasetErrorDetail(e);
      this.refusal = {
        code: d?.error ?? null,
        message: apiErrorText(e),
        importId: d?.import_id ?? importId,
        issues: d?.issues ?? [],
      };
      if (d?.error === 'dataset_changed') void this.runPreview();
      return null;
    }
  }

  // -- upload -------------------------------------------------------------

  /**
   * Upload an archive, then preview its served `dataset_path`. `maxBytes`
   * is the lower of the served `upload.max_bytes` and the proxy cap.
   */
  async upload(file: File, maxBytes: number, tooLargeMessage: string): Promise<void> {
    this.uploadError = null;
    if (file.size > maxBytes) {
      this.uploadError = tooLargeMessage;
      return;
    }
    this.uploading = true;
    try {
      const res = await this.#deps.uploadDatasetArchive(file);
      this.setSource(res.dataset_path);
      await this.runPreview();
    } catch (e) {
      this.uploadError = apiErrorText(e);
    } finally {
      this.uploading = false;
    }
  }

  destroy(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
    this.#abort?.abort();
  }
}

export function createImportWizard(deps: Partial<WizardDeps> = {}): ImportWizard {
  return new ImportWizard(deps);
}
