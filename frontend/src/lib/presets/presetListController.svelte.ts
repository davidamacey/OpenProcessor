/**
 * A preset kind's list page: the shared list state (`ConfigListBase`: the served list, the active
 * refs, Clone and Delete) plus create. Clone copies the served document under a new name with a
 * `PUT` that creates; so does create, with a body the operator wrote. The server validates both.
 */
import {
  deletePreset,
  getPreset,
  listPresets,
  presetRefusalText,
  putPreset,
} from '$lib/api_presets';
import { ConfigListBase } from '$lib/config/configList.svelte';
import type { PresetBody, PresetDoc, PresetListResponse } from '$lib/types_presets';
import { PresetActive, type PresetActiveDeps } from './presetActive.svelte';
import type { PresetKind } from './presetKinds';

export interface PresetListDeps extends PresetActiveDeps {
  listPresets: typeof listPresets;
  getPreset: typeof getPreset;
  putPreset: typeof putPreset;
  deletePreset: typeof deletePreset;
}

export class PresetList extends ConfigListBase<
  PresetListResponse,
  PresetDoc,
  PresetActive
> {
  createError = $state<string | null>(null);
  readonly kind: PresetKind;
  #put: typeof putPreset;

  constructor(kind: PresetKind, deps: Partial<PresetListDeps> = {}) {
    const put = deps.putPreset ?? putPreset;
    super(
      {
        list: () => (deps.listPresets ?? listPresets)(kind.id),
        clone: async (from, req) => {
          const source = await (deps.getPreset ?? getPreset)(kind.id, from);
          return put(kind.id, req.new_name, {
            body: source.body,
            description: req.description ?? source.description,
            expected_revision: null,
          });
        },
        remove: (name, revision) =>
          (deps.deletePreset ?? deletePreset)(kind.id, name, revision),
      },
      new PresetActive(kind, deps),
    );
    this.kind = kind;
    this.#put = put;
  }

  /** "New": the document with the body the operator wrote. The new doc, or null (the served
   *  refusal, with its reasons, is in `createError`). */
  async create(
    name: string,
    description: string,
    body: PresetBody,
  ): Promise<PresetDoc | null> {
    if (this.busy) return null;
    this.busy = true;
    this.createError = null;
    try {
      const doc = await this.#put(this.kind.id, name.trim(), {
        body,
        description: description.trim(),
        expected_revision: null,
      });
      await this.load();
      return doc;
    } catch (e) {
      this.createError = presetRefusalText(e);
      return null;
    } finally {
      this.busy = false;
    }
  }
}

export function createPresetList(
  kind: PresetKind,
  deps: Partial<PresetListDeps> = {},
): PresetList {
  return new PresetList(kind, deps);
}
