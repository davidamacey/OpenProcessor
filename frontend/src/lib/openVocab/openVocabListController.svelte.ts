/**
 * The open-vocabulary list page's state: the open-vocab binding of the
 * shared `ConfigList` (served sets and templates, the active set with
 * Rollback and Turn off, Clone and Delete) plus create-from-nothing (the
 * template list may be empty). A `config.changed axis=open_vocab` event is
 * a wake-up to re-read.
 */
import {
  cloneOpenVocab,
  createOpenVocab,
  deleteOpenVocab,
  getOpenVocabSchema,
  listOpenVocab,
} from '$lib/api_openVocab';
import { apiErrorText } from '$lib/api';
import { ConfigList } from '$lib/config/configList.svelte';
import { isConfigAxisEvent } from '$lib/config/validationIssues';
import { subscribeCurationEvents, type CurationEvent } from '$lib/sse';
import type {
  OpenVocabCloneRequest,
  OpenVocabDoc,
  OpenVocabList,
  OpenVocabVocabulary,
} from '$lib/types_openVocab';
import { OpenVocabActive, type OpenVocabActiveDeps } from './openVocabActive.svelte';

export const OPEN_VOCAB_AXIS = 'open_vocab';

export function isOpenVocabConfigEvent(e: CurationEvent): boolean {
  return isConfigAxisEvent(e, OPEN_VOCAB_AXIS);
}

export interface OpenVocabListDeps extends OpenVocabActiveDeps {
  listOpenVocab: typeof listOpenVocab;
  getOpenVocabSchema: typeof getOpenVocabSchema;
  createOpenVocab: typeof createOpenVocab;
  cloneOpenVocab: typeof cloneOpenVocab;
  deleteOpenVocab: typeof deleteOpenVocab;
  subscribe: (onEvent: (e: CurationEvent) => void) => { close(): void };
}

export class OpenVocabListState extends ConfigList<
  OpenVocabList,
  OpenVocabDoc,
  OpenVocabActive
> {
  createError = $state<string | null>(null);
  /** The served labels for the pass's closed value sets (from the schema);
   *  null while unread or when the read failed, and then nothing that needs
   *  a label is offered. */
  vocabulary = $state<OpenVocabVocabulary | null>(null);
  #create: typeof createOpenVocab;
  #schema: typeof getOpenVocabSchema;

  constructor(deps: Partial<OpenVocabListDeps> = {}) {
    super(
      {
        list: () => (deps.listOpenVocab ?? listOpenVocab)(),
        clone: (name, body) =>
          (deps.cloneOpenVocab ?? cloneOpenVocab)(name, {
            ...body,
            // The clone dialog only ever offers a stored set or a template.
            source: body.source as OpenVocabCloneRequest['source'],
          }),
        remove: (name, rev) => (deps.deleteOpenVocab ?? deleteOpenVocab)(name, rev),
        subscribe:
          deps.subscribe ??
          ((onEvent) => subscribeCurationEvents({ topic: 'config', onEvent })),
        isEvent: isOpenVocabConfigEvent,
      },
      new OpenVocabActive(deps),
    );
    this.#create = deps.createOpenVocab ?? createOpenVocab;
    this.#schema = deps.getOpenVocabSchema ?? getOpenVocabSchema;
  }

  override async load(): Promise<void> {
    await Promise.all([super.load(), this.#loadVocabulary()]);
  }

  async #loadVocabulary(): Promise<void> {
    try {
      this.vocabulary = (await this.#schema()).vocabulary;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.vocabulary = null;
    }
  }

  async deactivate(): Promise<boolean> {
    const ok = await this.active.deactivate();
    if (ok) await this.load();
    return ok;
  }

  /** "New set": an empty body, so the server's defaults fill it. The new
   *  doc, or null (the served refusal is in `createError`). */
  async create(name: string): Promise<OpenVocabDoc | null> {
    if (this.busy) return null;
    this.busy = true;
    this.createError = null;
    try {
      return await this.#create({ name: name.trim(), body: {} });
    } catch (e) {
      this.createError = apiErrorText(e);
      return null;
    } finally {
      this.busy = false;
    }
  }
}

export function createOpenVocabList(
  deps: Partial<OpenVocabListDeps> = {},
): OpenVocabListState {
  return new OpenVocabListState(deps);
}
