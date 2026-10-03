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
  listOpenVocab,
} from '$lib/api_openVocab';
import { configErrorText } from '$lib/api';
import { ConfigList } from '$lib/config/configList.svelte';
import { isConfigAxisEvent } from '$lib/config/validationIssues';
import { subscribeCurationEvents, type CurationEvent } from '$lib/sse';
import type {
  OpenVocabCloneRequest,
  OpenVocabDoc,
  OpenVocabList,
} from '$lib/types_openVocab';
import { OpenVocabActive, type OpenVocabActiveDeps } from './openVocabActive.svelte';

export const OPEN_VOCAB_AXIS = 'open_vocab';

export function isOpenVocabConfigEvent(e: CurationEvent): boolean {
  return isConfigAxisEvent(e, OPEN_VOCAB_AXIS);
}

export interface OpenVocabListDeps extends OpenVocabActiveDeps {
  listOpenVocab: typeof listOpenVocab;
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
  #create: typeof createOpenVocab;

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
      this.createError = configErrorText(e);
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
