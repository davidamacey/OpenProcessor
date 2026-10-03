/**
 * The project's active open-vocabulary set (`GET /open_vocab/active`, axis
 * `open_vocab`) and the writes that move it: activate, rollback and
 * deactivate. The shared `ConfigActive`; every write sends `expected_active`
 * as last read (`{name: null, revision: null}` when nothing is active).
 */
import {
  activateOpenVocab,
  deactivateOpenVocab,
  getActiveOpenVocab,
  rollbackOpenVocab,
} from '$lib/api_openVocab';
import { ConfigActive } from '$lib/config/configActive.svelte';
import type { OpenVocabActivateResponse } from '$lib/types_openVocab';

export interface OpenVocabActiveDeps {
  getActiveOpenVocab: typeof getActiveOpenVocab;
  activateOpenVocab: typeof activateOpenVocab;
  rollbackOpenVocab: typeof rollbackOpenVocab;
  deactivateOpenVocab: typeof deactivateOpenVocab;
}

export class OpenVocabActive extends ConfigActive<OpenVocabActivateResponse> {
  constructor(deps: Partial<OpenVocabActiveDeps> = {}) {
    super({
      getActive: () => (deps.getActiveOpenVocab ?? getActiveOpenVocab)(),
      activate: (name, body) => (deps.activateOpenVocab ?? activateOpenVocab)(name, body),
      rollback: (body) => (deps.rollbackOpenVocab ?? rollbackOpenVocab)(body),
      deactivate: (body) => (deps.deactivateOpenVocab ?? deactivateOpenVocab)(body),
    });
  }
}
