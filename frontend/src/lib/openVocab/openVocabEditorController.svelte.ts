/**
 * One open-vocabulary set's editor state: the open-vocab binding of the
 * shared `ConfigEditor` (draft, served live validation, Save with
 * `expected_revision`, revisions and restore, the `config.changed
 * axis=open_vocab` wake-up) plus the structured edits the body needs
 * (a list of targets, a gating object with a hit-rate sub-object) and
 * "Check for activation" (the draft posted with `for_activation=true`, its
 * report kept apart). New targets take the served rows' defaults; no
 * client rule checks a value.
 */
import { configErrorText } from '$lib/api';
import {
  getOpenVocab,
  getOpenVocabRevision,
  getOpenVocabRevisions,
  getOpenVocabSchema,
  listOpenVocab,
  updateOpenVocab,
  validateOpenVocab,
} from '$lib/api_openVocab';
import { ConfigEditor } from '$lib/config/configEditor.svelte';
import { subscribeCurationEvents, type CurationEvent } from '$lib/sse';
import type { ValidationReport } from '$lib/types_config';
import type {
  OpenVocabBody,
  OpenVocabDoc,
  OpenVocabGatingBody,
  OpenVocabHitRateBody,
  OpenVocabSchema,
  SegmenterAvailability,
  OpenVocabTargetBody,
} from '$lib/types_openVocab';
import { OpenVocabActive, type OpenVocabActiveDeps } from './openVocabActive.svelte';
import { defaultTarget } from './openVocabFields';
import { isOpenVocabConfigEvent } from './openVocabListController.svelte';

export interface OpenVocabEditorDeps extends OpenVocabActiveDeps {
  getOpenVocab: typeof getOpenVocab;
  getOpenVocabSchema: typeof getOpenVocabSchema;
  listOpenVocab: typeof listOpenVocab;
  getOpenVocabRevisions: typeof getOpenVocabRevisions;
  getOpenVocabRevision: typeof getOpenVocabRevision;
  updateOpenVocab: typeof updateOpenVocab;
  validateOpenVocab: typeof validateOpenVocab;
  subscribe: (onEvent: (e: CurationEvent) => void) => { close(): void };
}

export class OpenVocabEditor extends ConfigEditor<
  OpenVocabBody,
  OpenVocabDoc,
  OpenVocabSchema,
  OpenVocabActive
> {
  /** The served report of the last "Check for activation". */
  activationReport = $state<ValidationReport | null>(null);
  activationChecking = $state(false);
  activationCheckError = $state<string | null>(null);

  /** The served segmenter fact (`GET /open_vocab`); null while unread. */
  segmenter = $state<SegmenterAvailability | null>(null);

  #validate: typeof validateOpenVocab;
  #list: typeof listOpenVocab;

  constructor(name: string, deps: Partial<OpenVocabEditorDeps> = {}) {
    super(
      name,
      {
        getSchema: () => (deps.getOpenVocabSchema ?? getOpenVocabSchema)(),
        getDoc: (n) => (deps.getOpenVocab ?? getOpenVocab)(n),
        getRevisions: (n) => (deps.getOpenVocabRevisions ?? getOpenVocabRevisions)(n),
        getRevision: (n, r) => (deps.getOpenVocabRevision ?? getOpenVocabRevision)(n, r),
        update: (n, body) => (deps.updateOpenVocab ?? updateOpenVocab)(n, body),
        validate: (body, signal) =>
          (deps.validateOpenVocab ?? validateOpenVocab)(body, false, signal),
        subscribe:
          deps.subscribe ??
          ((onEvent) => subscribeCurationEvents({ topic: 'config', onEvent })),
        isEvent: isOpenVocabConfigEvent,
      },
      new OpenVocabActive(deps),
    );
    this.#validate = deps.validateOpenVocab ?? validateOpenVocab;
    this.#list = deps.listOpenVocab ?? listOpenVocab;
  }

  /** The segmenter fact rides on the list read; a failed read leaves it
   *  unread and never fails the editor's own load. */
  protected override async loadExtras(): Promise<void> {
    try {
      this.segmenter = (await this.#list()).segmenter;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.segmenter = null;
    }
  }

  #setTargets(targets: OpenVocabTargetBody[]): void {
    this.draftBody = { ...this.draftBody, targets };
    this.activationReport = null;
    this.scheduleValidate();
  }

  get targets(): OpenVocabTargetBody[] {
    return this.draftBody.targets ?? [];
  }

  override setField(field: string, value: OpenVocabBody[keyof OpenVocabBody]): void {
    super.setField(field, value as never);
    this.activationReport = null;
  }

  addTarget(): void {
    if (!this.editable || !this.schema) return;
    this.#setTargets([...this.targets, defaultTarget(this.schema)]);
  }

  removeTarget(index: number): void {
    if (!this.editable) return;
    this.#setTargets(this.targets.filter((_, i) => i !== index));
  }

  /** `dir` -1 moves up, +1 down; a move past either end does nothing. */
  moveTarget(index: number, dir: -1 | 1): void {
    const to = index + dir;
    if (!this.editable || to < 0 || to >= this.targets.length) return;
    const next = [...this.targets];
    [next[index], next[to]] = [next[to]!, next[index]!];
    this.#setTargets(next);
  }

  setTargetField(index: number, field: string, value: unknown): void {
    if (!this.editable || index < 0 || index >= this.targets.length) return;
    this.#setTargets(
      this.targets.map((t, i) => (i === index ? { ...t, [field]: value } : t)),
    );
  }

  setGatingField(field: string, value: unknown): void {
    if (!this.editable) return;
    const gating = { ...(this.draftBody.gating ?? {}), [field]: value };
    this.draftBody = { ...this.draftBody, gating: gating as OpenVocabGatingBody };
    this.activationReport = null;
    this.scheduleValidate();
  }

  setHitRateField(field: string, value: unknown): void {
    if (!this.editable) return;
    const hit = { ...(this.draftBody.gating?.tier3_hit_rate ?? {}), [field]: value };
    this.setGatingField('tier3_hit_rate', hit as OpenVocabHitRateBody);
  }

  /** Posts the draft with `for_activation=true`. Never blocks anything;
   *  the served report is shown apart from the live one. */
  async checkActivation(): Promise<void> {
    this.activationChecking = true;
    try {
      this.activationReport = await this.#validate(
        { name: null, body: this.draftBody },
        true,
      );
      this.activationCheckError = null;
    } catch (e) {
      this.activationCheckError = configErrorText(e);
    } finally {
      this.activationChecking = false;
    }
  }
}

export function createOpenVocabEditor(
  name: string,
  deps: Partial<OpenVocabEditorDeps> = {},
): OpenVocabEditor {
  return new OpenVocabEditor(name, deps);
}
