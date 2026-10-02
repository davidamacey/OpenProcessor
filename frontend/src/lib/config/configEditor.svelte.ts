/**
 * One config doc's editor state, shared by the prompt-pack (W3) and
 * region-profile (W4) editors (any_domain_plan.md §3.5, §4.4, §7.2, §7.3,
 * §7.6 items 1, 2 and 4).
 *
 * The draft is the served body plus the operator's edits. Validation is
 * the server's: 400 ms after an edit the draft is posted to the resource's
 * `/validate` (the previous request aborted) and the served report
 * replaces the one on screen. Nothing here checks a field. Save sends
 * `expected_revision` = the loaded revision; a 409 `revision_conflict`
 * offers "reload" or "keep my edits" (adopting the served
 * `current_revision`). A `config.changed` event on the resource's axis
 * re-reads the active ref, and the doc when it names this one and the
 * draft is clean (a dirty draft gets a notice instead).
 */
import { configErrorDetail, configErrorText } from '$lib/api';
import type { CurationEvent } from '$lib/sse';
import type {
  ActiveConfigResponse,
  ConfigDocBase,
  ConfigRevision,
  ConfigRevisionList,
  ConfigUpdateRequest,
  ConfigValidateRequest,
  ValidationReport,
} from '$lib/types_config';
import type { ConfigActive } from './configActive.svelte';

export const VALIDATE_DEBOUNCE_MS = 400;

export type ConfigBody = Record<string, unknown>;

/** The resource's routes, already bound to its API wrappers. */
export interface ConfigEditorBackend<
  B extends ConfigBody,
  D extends ConfigDocBase<B>,
  S,
> {
  getSchema: () => Promise<S>;
  getDoc: (name: string) => Promise<D>;
  getRevisions: (name: string) => Promise<ConfigRevisionList>;
  getRevision: (name: string, revision: number) => Promise<D>;
  update: (name: string, body: ConfigUpdateRequest<B>) => Promise<D>;
  validate: (
    body: ConfigValidateRequest<B>,
    signal: AbortSignal,
  ) => Promise<ValidationReport>;
  subscribe: (onEvent: (e: CurationEvent) => void) => { close(): void };
  /** True for the `config.changed` events this resource follows. */
  isEvent: (e: CurationEvent) => boolean;
}

/** What the shared editor components (save panel, revisions, activate and
 *  restore dialogs) read and call, whatever the resource. */
export interface ConfigEditorView {
  readonly name: string;
  readonly active: ConfigActive<ActiveConfigResponse>;
  readonly doc: ConfigDocBase<unknown> | null;
  readonly viewing: ConfigDocBase<unknown> | null;
  readonly revisions: ConfigRevision[] | null;
  readonly revisionError: string | null;
  readonly report: ValidationReport | null;
  readonly validating: boolean;
  readonly validateError: string | null;
  readonly saving: boolean;
  readonly saveError: string | null;
  readonly conflict: { message: string; currentRevision: number | null } | null;
  readonly remoteChanged: boolean;
  readonly dirty: boolean;
  readonly editable: boolean;
  readonly canSave: boolean;
  save(): Promise<boolean>;
  reloadLatest(): Promise<void>;
  keepMine(): void;
  viewRevision(revision: number): Promise<void>;
  closeRevision(): void;
  restoreViewed(): Promise<boolean>;
}

/** A deep copy, so edits never touch the served doc. */
export function cloneBody<B>(body: B): B {
  return JSON.parse(JSON.stringify(body)) as B;
}

export class ConfigEditor<
  B extends ConfigBody,
  D extends ConfigDocBase<B>,
  S,
  A extends { load(): Promise<void> } = ConfigActive,
> {
  readonly name: string;
  readonly active: A;

  schema = $state<S | null>(null);
  doc = $state<D | null>(null);
  revisions = $state<ConfigRevision[] | null>(null);
  loadError = $state<string | null>(null);

  draftBody = $state<B>({} as B);
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
  viewing = $state<D | null>(null);
  revisionError = $state<string | null>(null);
  /** This doc changed on the server while the draft had edits. */
  remoteChanged = $state(false);

  protected backend: ConfigEditorBackend<B, D, S>;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #validateAbort: AbortController | null = null;
  #sub: { close(): void } | null = null;

  constructor(name: string, backend: ConfigEditorBackend<B, D, S>, active: A) {
    this.name = name;
    this.backend = backend;
    this.active = active;
  }

  get dirty(): boolean {
    const d = this.doc;
    if (!d) return false;
    return (
      JSON.stringify(this.draftBody) !== JSON.stringify(d.body) ||
      this.draftDescription !== (d.description ?? '')
    );
  }

  /** A stored doc at its latest revision, not a revision being viewed. */
  get editable(): boolean {
    return !!this.doc && !this.doc.read_only && this.viewing == null;
  }

  get canSave(): boolean {
    return this.editable && this.dirty && !this.saving && this.expectedRevision != null;
  }

  protected adopt(doc: D): void {
    this.doc = doc;
    this.draftBody = cloneBody(doc.body);
    this.draftDescription = doc.description ?? '';
    this.expectedRevision = doc.revision;
    this.report = doc.validation;
    this.conflict = null;
    this.remoteChanged = false;
  }

  /** Extra reads a resource's editor needs alongside the doc (a profile
   *  editor's vocabulary). Runs in parallel with the doc read. */
  protected async loadExtras(): Promise<void> {}

  async load(): Promise<void> {
    const b = this.backend;
    try {
      const [schema, doc] = await Promise.all([
        b.getSchema(),
        b.getDoc(this.name),
        this.active.load(),
        this.loadExtras(),
      ]);
      this.schema = schema;
      this.adopt(doc);
      this.loadError = null;
      await this.loadRevisions();
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.loadError = configErrorText(e);
    }
  }

  async loadRevisions(): Promise<void> {
    if (!this.doc || this.doc.source !== 'stored') {
      this.revisions = null;
      return;
    }
    try {
      this.revisions = (await this.backend.getRevisions(this.name)).revisions;
      this.revisionError = null;
    } catch (e) {
      this.revisionError = configErrorText(e);
    }
  }

  start(): void {
    void this.load();
    this.#sub = this.backend.subscribe((e) => void this.#onEvent(e));
  }

  stop(): void {
    this.#sub?.close();
    this.#sub = null;
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
    this.#validateAbort?.abort();
  }

  async #onEvent(e: CurationEvent): Promise<void> {
    if (!this.backend.isEvent(e)) return;
    await this.active.load();
    const name = (e as { name?: string | null }).name;
    if (name !== this.name) return;
    if (this.dirty) {
      this.remoteChanged = true;
      return;
    }
    try {
      this.adopt(await this.backend.getDoc(this.name));
      await this.loadRevisions();
    } catch (e2) {
      this.loadError = configErrorText(e2);
    }
  }

  setField(field: string, value: B[string]): void {
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

  /** Posts the draft to `/validate`. `name: null`: the doc's own name
   *  would read as taken (W3-Q9, W4-Q14). */
  async validateNow(): Promise<void> {
    this.#validateAbort?.abort();
    const ctl = new AbortController();
    this.#validateAbort = ctl;
    this.validating = true;
    try {
      const report = await this.backend.validate(
        { name: null, body: this.draftBody },
        ctl.signal,
      );
      if (ctl.signal.aborted) return;
      this.report = report;
      this.validateError = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.validateError = configErrorText(e);
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

  async #put(body: B, description: string): Promise<boolean> {
    if (this.expectedRevision == null) return false;
    this.saving = true;
    this.saveError = null;
    this.conflict = null;
    try {
      const doc = await this.backend.update(this.name, {
        expected_revision: this.expectedRevision,
        description: description.trim() || null,
        body,
      });
      this.viewing = null;
      this.adopt(doc);
      await this.loadRevisions();
      return true;
    } catch (e) {
      const d = configErrorDetail(e);
      if (d?.error === 'revision_conflict') {
        this.conflict = {
          message: d.message,
          currentRevision: d.current_revision ?? null,
        };
      } else {
        this.saveError = configErrorText(e);
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
      this.adopt(await this.backend.getDoc(this.name));
      await this.loadRevisions();
    } catch (e) {
      this.loadError = configErrorText(e);
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
      this.viewing = await this.backend.getRevision(this.name, revision);
    } catch (e) {
      this.revisionError = configErrorText(e);
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
