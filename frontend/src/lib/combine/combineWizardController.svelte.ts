/**
 * The combine-projects wizard's state (projects_plan.md §6, §8 "Combine
 * wizard"; docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md §4.3).
 *
 * Holds only what the operator chose (sources in priority order, the
 * target, explicit mapping rows, touched options) and what the server
 * answered (the preview, the start outcome). Every count, error, warning
 * and target class is the served preview's; the client decides none of
 * them. An option the operator never touched is omitted from the body, so
 * the server's own default applies.
 */
import { getDatasetFormatsFor, previewCombine, startCombine } from '$lib/api_combine';
import { configErrorDetail, configErrorText } from '$lib/api';
import type {
  CombineDedupMode,
  CombineHoldoutMode,
  CombineIssue,
  CombineLabelStates,
  CombineMappingActionChoice,
  CombineMappingEntry,
  CombinePreview,
  CombineRequest,
  CombineSource,
  CombineStartResponse,
} from '$lib/types_combine';
import type { ConfigErrorJobRef, ValidationReport } from '$lib/types_config';

export const COMBINE_PREVIEW_DEBOUNCE_MS = 400;

/** The four `ClassMappingEntry` actions (contract enum). */
export const COMBINE_MAPPING_ACTIONS = ['map', 'create', 'skip', 'region'] as const;

/** One mapping row. `action: ''` = not chosen yet (row omitted). */
export interface CombineMappingChoice {
  action: string;
  /** `create`: the class this row defines (empty = the source class name);
   *  `map`: the created class it maps onto. */
  new_class_name: string;
  /** The operator edited this row; a re-preview never overwrites it with
   *  the served suggestion. */
  touched: boolean;
}

export interface CombineSourceChoice {
  project: string;
  /** `undefined` until the operator picks one (the server's default). */
  label_states?: CombineLabelStates;
}

export interface CombineOptionChoices {
  dedup?: CombineDedupMode;
  dedup_iou?: number;
  holdout?: CombineHoldoutMode;
  settings_from?: string;
}

export interface CombineStartRefusal {
  code: string | null;
  message: string;
  report: ValidationReport | null;
  jobs: ConfigErrorJobRef[];
}

export interface CombineWizardDeps {
  previewCombine: typeof previewCombine;
  startCombine: typeof startCombine;
  getDatasetFormatsFor: typeof getDatasetFormatsFor;
  /** The served project to read mapping labels from; `null` = unknown. */
  projectOf: (slug: string) => { prefix: string } | null;
}

const DEFAULT_DEPS: CombineWizardDeps = {
  previewCombine,
  startCombine,
  getDatasetFormatsFor,
  projectOf: () => null,
};

export class CombineWizard {
  sources = $state<CombineSourceChoice[]>([]);
  slug = $state('');
  displayName = $state('');
  description = $state('');
  options = $state<CombineOptionChoices>({});
  /** project slug -> source class name -> choice. */
  choices = $state<Record<string, Record<string, CombineMappingChoice>>>({});

  preview = $state<CombinePreview | null>(null);
  previewError = $state<string | null>(null);
  previewing = $state(false);
  /** The request body (JSON) the preview on screen was served for. */
  servedKey = $state<string | null>(null);

  starting = $state(false);
  refusal = $state<CombineStartRefusal | null>(null);

  /** Served `mapping_actions` labels, `null` until read (or when the read
   *  404'd: raw ids are shown). */
  actionChoices = $state<CombineMappingActionChoice[] | null>(null);

  #deps: CombineWizardDeps;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #abort: AbortController | null = null;
  #seq = 0;
  #labelsFor: string | null = null;

  constructor(deps: Partial<CombineWizardDeps> = {}) {
    this.#deps = { ...DEFAULT_DEPS, ...deps };
  }

  // -- inputs -------------------------------------------------------------

  setTarget(patch: { slug?: string; displayName?: string; description?: string }): void {
    if (patch.slug !== undefined) this.slug = patch.slug;
    if (patch.displayName !== undefined) this.displayName = patch.displayName;
    if (patch.description !== undefined) this.description = patch.description;
    this.#changed();
  }

  hasSource(slug: string): boolean {
    return this.sources.some((s) => s.project === slug);
  }

  addSource(slug: string): void {
    if (this.hasSource(slug)) return;
    this.sources = [...this.sources, { project: slug }];
    this.#syncSources();
    this.#changed();
  }

  removeSource(slug: string): void {
    this.sources = this.sources.filter((s) => s.project !== slug);
    const { [slug]: _gone, ...rest } = this.choices;
    this.choices = rest;
    if (this.options.settings_from === slug) this.setOption('settings_from', undefined);
    this.#syncSources();
    this.#changed();
  }

  /** Priority order: the first source wins a label conflict. */
  moveSource(slug: string, delta: -1 | 1): void {
    const i = this.sources.findIndex((s) => s.project === slug);
    const j = i + delta;
    if (i < 0 || j < 0 || j >= this.sources.length) return;
    const next = [...this.sources];
    [next[i], next[j]] = [next[j]!, next[i]!];
    this.sources = next;
    this.#syncSources();
    this.#changed();
  }

  setLabelStates(slug: string, value: CombineLabelStates): void {
    this.sources = this.sources.map((s) =>
      s.project === slug ? { ...s, label_states: value } : s,
    );
    this.#changed();
  }

  setOption<K extends keyof CombineOptionChoices>(
    key: K,
    value: CombineOptionChoices[K] | undefined,
  ): void {
    const next = { ...this.options };
    if (value === undefined) delete next[key];
    else next[key] = value;
    this.options = next;
    this.#changed();
  }

  choiceFor(project: string, datasetClass: string): CombineMappingChoice {
    return (
      this.choices[project]?.[datasetClass] ?? {
        action: '',
        new_class_name: '',
        touched: false,
      }
    );
  }

  /** An operator edit of one row. */
  setChoice(
    project: string,
    datasetClass: string,
    patch: Partial<Pick<CombineMappingChoice, 'action' | 'new_class_name'>>,
  ): void {
    const prev = this.choiceFor(project, datasetClass);
    const next: CombineMappingChoice = { ...prev, ...patch, touched: true };
    if (patch.action !== undefined && patch.action !== prev.action) {
      // The name field means something different per action.
      if (patch.action === 'create' && patch.new_class_name === undefined)
        next.new_class_name = datasetClass;
      else if (patch.action !== 'create' && patch.action !== 'map')
        next.new_class_name = '';
      else if (patch.action === 'map' && patch.new_class_name === undefined)
        next.new_class_name = '';
    }
    this.choices = {
      ...this.choices,
      [project]: { ...(this.choices[project] ?? {}), [datasetClass]: next },
    };
    this.#changed();
  }

  /** The class names the form's own `create` rows currently define,
   *  across every source, in first-appearance order. */
  get createdNames(): string[] {
    const out: string[] = [];
    for (const s of this.sources) {
      for (const [cls, c] of Object.entries(this.choices[s.project] ?? {})) {
        if (c.action !== 'create') continue;
        const name = c.new_class_name.trim() || cls;
        if (!out.includes(name)) out.push(name);
      }
    }
    return out;
  }

  /** "Reset to suggestions": drop every touched flag and re-copy the
   *  served suggestions. */
  resetToSuggestions(): void {
    const next: Record<string, Record<string, CombineMappingChoice>> = {};
    for (const [p, rows] of Object.entries(this.choices)) {
      next[p] = Object.fromEntries(
        Object.entries(rows).map(([k, v]) => [k, { ...v, touched: false }]),
      );
    }
    this.choices = next;
    if (this.preview) this.#applySuggestions(this.preview);
    this.#changed();
  }

  /** Untouched rows take the served suggestion; touched rows never do.
   *  Returns whether any row changed. */
  #applySuggestions(p: CombinePreview): boolean {
    let changed = false;
    const next = { ...this.choices };
    for (const s of this.sources) {
      const entries = p.suggested_mapping?.[s.project] ?? [];
      for (const e of entries) {
        const cur = next[s.project]?.[e.dataset_class];
        if (cur?.touched) continue;
        const name = e.new_class_name ?? '';
        if (cur && cur.action === e.action && cur.new_class_name === name) continue;
        next[s.project] = {
          ...(next[s.project] ?? {}),
          [e.dataset_class]: { action: e.action, new_class_name: name, touched: false },
        };
        changed = true;
      }
    }
    if (changed) this.choices = next;
    return changed;
  }

  // -- request ------------------------------------------------------------

  #mappingEntries(project: string): CombineMappingEntry[] {
    const out: CombineMappingEntry[] = [];
    for (const [dataset_class, c] of Object.entries(this.choices[project] ?? {})) {
      if (!c.action) continue;
      const entry: CombineMappingEntry = { dataset_class, action: c.action };
      if (c.action === 'create' || c.action === 'map') {
        if (c.new_class_name.trim()) entry.new_class_name = c.new_class_name.trim();
      }
      out.push(entry);
    }
    return out;
  }

  requestBody(): CombineRequest {
    const sources: CombineSource[] = this.sources.map((s) =>
      s.label_states
        ? { project: s.project, include: { label_states: s.label_states } }
        : { project: s.project },
    );
    const class_mapping: Record<string, CombineMappingEntry[]> = {};
    for (const s of this.sources) {
      const rows = this.#mappingEntries(s.project);
      if (rows.length > 0) class_mapping[s.project] = rows;
    }
    const target: CombineRequest['target'] = {
      slug: this.slug.trim(),
      display_name: this.displayName.trim(),
    };
    if (this.description.trim()) target.description = this.description.trim();
    const body: CombineRequest = { target, sources };
    if (Object.keys(class_mapping).length > 0) body.class_mapping = class_mapping;
    const o = this.options;
    if (o.dedup !== undefined) body.dedup = o.dedup;
    if (o.dedup_iou !== undefined) body.dedup_iou = o.dedup_iou;
    if (o.holdout !== undefined) body.holdout = o.holdout;
    if (o.settings_from !== undefined) body.settings_from = o.settings_from;
    return body;
  }

  /** The preview needs a source and the target's two required fields. */
  get canPreview(): boolean {
    return (
      this.sources.length > 0 && this.slug.trim() !== '' && this.displayName.trim() !== ''
    );
  }

  /** The request changed since the preview on screen was served. */
  get stale(): boolean {
    return (
      this.servedKey !== null && this.servedKey !== JSON.stringify(this.requestBody())
    );
  }

  get canStart(): boolean {
    return (
      this.preview?.ok === true &&
      this.previewError === null &&
      !this.previewing &&
      !this.stale &&
      !this.starting
    );
  }

  // -- preview ------------------------------------------------------------

  #changed(): void {
    this.refusal = null;
    this.#schedulePreview();
  }

  #schedulePreview(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = setTimeout(() => {
      this.#timer = null;
      void this.runPreview();
    }, COMBINE_PREVIEW_DEBOUNCE_MS);
  }

  async runPreview(): Promise<void> {
    if (this.#timer) {
      clearTimeout(this.#timer);
      this.#timer = null;
    }
    this.#abort?.abort();
    if (!this.canPreview) {
      this.#abort = null;
      this.#seq += 1;
      this.preview = null;
      this.previewError = null;
      this.previewing = false;
      this.servedKey = null;
      return;
    }
    const ctrl = new AbortController();
    this.#abort = ctrl;
    const seq = ++this.#seq;
    const body = this.requestBody();
    this.previewing = true;
    try {
      const p = await this.#deps.previewCombine(body, ctrl.signal);
      if (seq !== this.#seq) return;
      this.preview = p;
      this.previewError = null;
      this.servedKey = JSON.stringify(body);
      // The suggestions are the preview's; copy them into untouched rows,
      // then re-preview once with them in the request.
      if (this.#applySuggestions(p)) this.#schedulePreview();
    } catch (e) {
      if ((e as Error)?.name === 'AbortError' || seq !== this.#seq) return;
      this.previewError = configErrorText(e);
      this.preview = null;
      this.servedKey = null;
    } finally {
      if (seq === this.#seq) this.previewing = false;
    }
  }

  // -- start --------------------------------------------------------------

  /** Returns the started job, or `null` when the server refused. */
  async start(): Promise<CombineStartResponse | null> {
    const p = this.preview;
    if (!p || this.starting) return null;
    this.starting = true;
    this.refusal = null;
    try {
      return await this.#deps.startCombine({
        ...this.requestBody(),
        expected_preview_sha: p.preview_sha,
      });
    } catch (e) {
      const d = configErrorDetail(e);
      this.refusal = {
        code: d?.error ?? null,
        message: configErrorText(e),
        report: d?.report ?? null,
        jobs: d?.jobs ?? [],
      };
      if (d?.error === 'preview_stale') void this.runPreview();
      return null;
    } finally {
      this.starting = false;
    }
  }

  // -- vocabulary ---------------------------------------------------------

  /** Label of a mapping action: the served one, else the raw id. */
  actionLabel(action: string): string {
    return this.actionChoices?.find((c) => c.value === action)?.label ?? action;
  }

  actionDescription(action: string): string | null {
    return this.actionChoices?.find((c) => c.value === action)?.description || null;
  }

  /** Read the labels from the first source's `/datasets/formats` once per
   *  first source; a failed read keeps the raw ids. */
  #syncSources(): void {
    const first = this.sources[0]?.project ?? null;
    if (first === this.#labelsFor) return;
    this.#labelsFor = first;
    this.actionChoices = null;
    const project = first ? this.#deps.projectOf(first) : null;
    if (!project) return;
    void this.#deps
      .getDatasetFormatsFor(project)
      .then((f) => {
        if (this.#labelsFor === first) this.actionChoices = f.mapping_actions ?? null;
      })
      .catch(() => {
        /* 404 (no W10) or a failed read: the raw ids render. */
      });
  }

  /** The preview's issues with a project, for the mapping tables. */
  issuesFor(project: string): CombineIssue[] {
    const p = this.preview;
    if (!p) return [];
    return [...p.errors, ...p.warnings].filter((i) => i.project === project);
  }

  destroy(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
    this.#abort?.abort();
  }
}

export function createCombineWizard(
  deps: Partial<CombineWizardDeps> = {},
): CombineWizard {
  return new CombineWizard(deps);
}
