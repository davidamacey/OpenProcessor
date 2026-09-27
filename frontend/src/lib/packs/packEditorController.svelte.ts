/**
 * One prompt pack's editor state (any_domain_plan.md §3.3, §3.5, §7.2,
 * §7.6 items 1 and 4; docs/design/w3-pack-editor-ui-plan-2026-09-27.md §3).
 *
 * The draft is the served body plus the operator's edits. Validation is
 * the server's: 400 ms after an edit the draft is posted to
 * `/prompt_packs/validate` (the previous request aborted) and the served
 * report replaces the one on screen. Nothing here checks a placeholder,
 * a reply key, a name or a length. Save sends `expected_revision` = the
 * loaded revision; a 409 `revision_conflict` offers "reload" or "keep my
 * edits" (adopting the served `current_revision`).
 */
import {
  getPromptPack,
  getPromptPackRevision,
  getPromptPackRevisions,
  getPromptPackSchema,
  packErrorDetail,
  packErrorText,
  updatePromptPack,
  validatePromptPack,
} from '$lib/api';
import { subscribeCurationEvents, type CurationEvent } from '$lib/sse';
import type {
  PackFieldValue,
  PromptPackBody,
  PromptPackDoc,
  PromptPackRevision,
  PromptPackSchema,
  ValidationIssue,
  ValidationReport,
} from '$lib/types_packs';
import { PackActive, type PackActiveDeps } from './packActive.svelte';
import { isPackConfigEvent } from './packListController.svelte';

export const VALIDATE_DEBOUNCE_MS = 400;

export interface PackEditorDeps extends PackActiveDeps {
  getPromptPack: typeof getPromptPack;
  getPromptPackSchema: typeof getPromptPackSchema;
  getPromptPackRevisions: typeof getPromptPackRevisions;
  getPromptPackRevision: typeof getPromptPackRevision;
  updatePromptPack: typeof updatePromptPack;
  validatePromptPack: typeof validatePromptPack;
  subscribe: (onEvent: (e: CurationEvent) => void) => { close(): void };
}

const DEFAULT_DEPS: Omit<PackEditorDeps, keyof PackActiveDeps> = {
  getPromptPack,
  getPromptPackSchema,
  getPromptPackRevisions,
  getPromptPackRevision,
  updatePromptPack,
  validatePromptPack,
  subscribe: (onEvent) => subscribeCurationEvents({ topic: 'config', onEvent }),
};

/** A deep copy, so edits never touch the served doc. */
function cloneBody(body: PromptPackBody): PromptPackBody {
  return JSON.parse(JSON.stringify(body)) as PromptPackBody;
}

/** The served issues whose `field` path names `field` (the path itself,
 *  or a dotted path under it, e.g. a map entry). */
export function issuesForField(
  report: ValidationReport | null,
  field: string,
): ValidationIssue[] {
  if (!report) return [];
  return [...report.errors, ...report.warnings].filter(
    (i) => i.field === field || (i.field ?? '').startsWith(`${field}.`),
  );
}

/** The served issues no schema field claims (whole-body, or a path the
 *  schema doesn't list). */
export function unplacedIssues(
  report: ValidationReport | null,
  fields: string[],
): ValidationIssue[] {
  if (!report) return [];
  return [...report.errors, ...report.warnings].filter(
    (i) =>
      i.field == null ||
      !fields.some((f) => i.field === f || i.field!.startsWith(`${f}.`)),
  );
}

export class PackEditor {
  readonly name: string;
  readonly active: PackActive;

  schema = $state<PromptPackSchema | null>(null);
  doc = $state<PromptPackDoc | null>(null);
  revisions = $state<PromptPackRevision[] | null>(null);
  loadError = $state<string | null>(null);

  draftBody = $state<PromptPackBody>({});
  draftDescription = $state('');
  /** The revision the next Save claims to replace. */
  expectedRevision = $state<number | null>(null);

  /** The report on screen: the doc's, then each live validation's. */
  report = $state<ValidationReport | null>(null);
  validating = $state(false);
  validateError = $state<string | null>(null);

  saving = $state(false);
  saveError = $state<string | null>(null);
  /** A 409 `revision_conflict` on Save: the served message and revision. */
  conflict = $state<{ message: string; currentRevision: number | null } | null>(null);

  /** A revision opened read-only from the history. */
  viewing = $state<PromptPackDoc | null>(null);
  revisionError = $state<string | null>(null);
  /** This pack changed on the server while the draft had edits. */
  remoteChanged = $state(false);

  #deps: PackEditorDeps;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #validateAbort: AbortController | null = null;
  #sub: { close(): void } | null = null;

  constructor(name: string, deps: Partial<PackEditorDeps> = {}) {
    this.name = name;
    this.#deps = { ...DEFAULT_DEPS, ...deps } as PackEditorDeps;
    this.active = new PackActive(deps);
  }

  get dirty(): boolean {
    const d = this.doc;
    if (!d) return false;
    return (
      JSON.stringify(this.draftBody) !== JSON.stringify(d.body) ||
      this.draftDescription !== (d.description ?? '')
    );
  }

  /** A stored pack at its latest revision, not a revision being viewed. */
  get editable(): boolean {
    return !!this.doc && !this.doc.read_only && this.viewing == null;
  }

  get canSave(): boolean {
    return this.editable && this.dirty && !this.saving && this.expectedRevision != null;
  }

  #adopt(doc: PromptPackDoc): void {
    this.doc = doc;
    this.draftBody = cloneBody(doc.body);
    this.draftDescription = doc.description ?? '';
    this.expectedRevision = doc.revision;
    this.report = doc.validation;
    this.conflict = null;
    this.remoteChanged = false;
  }

  async load(): Promise<void> {
    const d = this.#deps;
    try {
      const [schema, doc] = await Promise.all([
        d.getPromptPackSchema(),
        d.getPromptPack(this.name),
        this.active.load(),
      ]);
      this.schema = schema;
      this.#adopt(doc);
      this.loadError = null;
      await this.loadRevisions();
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.loadError = packErrorText(e);
    }
  }

  async loadRevisions(): Promise<void> {
    if (!this.doc || this.doc.source !== 'stored') {
      this.revisions = null;
      return;
    }
    try {
      this.revisions = (await this.#deps.getPromptPackRevisions(this.name)).revisions;
      this.revisionError = null;
    } catch (e) {
      this.revisionError = packErrorText(e);
    }
  }

  start(): void {
    void this.load();
    this.#sub = this.#deps.subscribe((e) => void this.#onEvent(e));
  }

  stop(): void {
    this.#sub?.close();
    this.#sub = null;
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
    this.#validateAbort?.abort();
  }

  async #onEvent(e: CurationEvent): Promise<void> {
    if (!isPackConfigEvent(e)) return;
    await this.active.load();
    const name = (e as { name?: string | null }).name;
    if (name !== this.name) return;
    if (this.dirty) {
      this.remoteChanged = true;
      return;
    }
    try {
      this.#adopt(await this.#deps.getPromptPack(this.name));
      await this.loadRevisions();
    } catch (e2) {
      this.loadError = packErrorText(e2);
    }
  }

  setField(field: string, value: PackFieldValue): void {
    if (!this.editable) return;
    this.draftBody = { ...this.draftBody, [field]: value };
    this.scheduleValidate();
  }

  setDescription(value: string): void {
    if (!this.editable) return;
    this.draftDescription = value;
  }

  scheduleValidate(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = setTimeout(() => {
      this.#timer = null;
      void this.validateNow();
    }, VALIDATE_DEBOUNCE_MS);
  }

  /** Posts the draft to `/validate`. `name: null`: the pack's own name
   *  would read as taken (W3-Q9). */
  async validateNow(): Promise<void> {
    this.#validateAbort?.abort();
    const ctl = new AbortController();
    this.#validateAbort = ctl;
    this.validating = true;
    try {
      const report = await this.#deps.validatePromptPack(
        { name: null, body: this.draftBody },
        ctl.signal,
      );
      if (ctl.signal.aborted) return;
      this.report = report;
      this.validateError = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.validateError = packErrorText(e);
    } finally {
      if (this.#validateAbort === ctl) {
        this.validating = false;
        this.#validateAbort = null;
      }
    }
  }

  async save(): Promise<boolean> {
    if (!this.canSave) return false;
    return this.#put(this.draftBody, this.draftDescription);
  }

  async #put(body: PromptPackBody, description: string): Promise<boolean> {
    if (this.expectedRevision == null) return false;
    this.saving = true;
    this.saveError = null;
    this.conflict = null;
    try {
      const doc = await this.#deps.updatePromptPack(this.name, {
        expected_revision: this.expectedRevision,
        description: description.trim() || null,
        body,
      });
      this.viewing = null;
      this.#adopt(doc);
      await this.loadRevisions();
      return true;
    } catch (e) {
      const d = packErrorDetail(e);
      if (d?.error === 'revision_conflict') {
        this.conflict = {
          message: d.message,
          currentRevision: d.current_revision ?? null,
        };
      } else {
        this.saveError = packErrorText(e);
        if (d?.report) this.report = d.report;
      }
      return false;
    } finally {
      this.saving = false;
    }
  }

  /** Conflict: drop the edits and load what the server has. */
  async reloadLatest(): Promise<void> {
    try {
      this.viewing = null;
      this.#adopt(await this.#deps.getPromptPack(this.name));
      await this.loadRevisions();
    } catch (e) {
      this.loadError = packErrorText(e);
    }
  }

  /** Conflict: keep the edits; the next Save claims the served revision. */
  keepMine(): void {
    if (!this.conflict || this.conflict.currentRevision == null) return;
    this.expectedRevision = this.conflict.currentRevision;
    this.conflict = null;
  }

  async viewRevision(revision: number): Promise<void> {
    this.revisionError = null;
    try {
      this.viewing = await this.#deps.getPromptPackRevision(this.name, revision);
    } catch (e) {
      this.revisionError = packErrorText(e);
    }
  }

  closeRevision(): void {
    this.viewing = null;
  }

  /** Saves the viewed revision's body and description as a new revision
   *  (§7.6 item 1 "restore"). */
  async restoreViewed(): Promise<boolean> {
    const v = this.viewing;
    if (!v || !this.doc || this.doc.read_only) return false;
    return this.#put(cloneBody(v.body), v.description ?? '');
  }
}

export function createPackEditor(
  name: string,
  deps: Partial<PackEditorDeps> = {},
): PackEditor {
  return new PackEditor(name, deps);
}
