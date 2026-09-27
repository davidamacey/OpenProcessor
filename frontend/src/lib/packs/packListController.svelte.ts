/**
 * The prompt-pack list page's state (any_domain_plan.md §7.6 items 1 and
 * 4; docs/design/w3-pack-editor-ui-plan-2026-09-27.md §2): the served
 * list, the active pack (with Rollback), Clone and Delete.
 *
 * A `config.changed` event on the `prompt_pack` axis is a wake-up to
 * re-read; the served list stays the only source.
 */
import {
  clonePromptPack,
  deletePromptPack,
  listPromptPacks,
  packErrorDetail,
  packErrorText,
} from '$lib/api';
import { subscribeCurationEvents, type CurationEvent } from '$lib/sse';
import type {
  PackSource,
  PromptPackDoc,
  PromptPackList,
  PromptPackSummary,
  ValidationReport,
} from '$lib/types_packs';
import { PackActive, type PackActiveDeps } from './packActive.svelte';

export interface PackListDeps extends PackActiveDeps {
  listPromptPacks: typeof listPromptPacks;
  clonePromptPack: typeof clonePromptPack;
  deletePromptPack: typeof deletePromptPack;
  subscribe: (onEvent: (e: CurationEvent) => void) => { close(): void };
}

const DEFAULT_SUBSCRIBE: PackListDeps['subscribe'] = (onEvent) =>
  subscribeCurationEvents({ topic: 'config', onEvent });

/** True for the `config.changed` events the pack surfaces follow. */
export function isPackConfigEvent(e: CurationEvent): boolean {
  return e.type === 'config.changed' && (e as { axis?: string }).axis === 'prompt_pack';
}

/** What the clone dialog clones: a pack, or a template (`source`). */
export interface CloneSource {
  name: string;
  source: PackSource | null;
}

export class PackList {
  list = $state<PromptPackList | null>(null);
  loadError = $state<string | null>(null);
  readonly active: PackActive;

  busy = $state(false);
  deleteError = $state<string | null>(null);
  cloneError = $state<string | null>(null);
  cloneReport = $state<ValidationReport | null>(null);

  #deps: Partial<PackListDeps>;
  #sub: { close(): void } | null = null;

  constructor(deps: Partial<PackListDeps> = {}) {
    this.#deps = deps;
    this.active = new PackActive(deps);
  }

  async load(): Promise<void> {
    const list = this.#deps.listPromptPacks ?? listPromptPacks;
    await Promise.all([
      (async () => {
        try {
          this.list = await list();
          this.loadError = null;
        } catch (e) {
          if ((e as Error)?.name === 'AbortError') return;
          this.loadError = packErrorText(e);
        }
      })(),
      this.active.load(),
    ]);
  }

  start(): void {
    void this.load();
    const subscribe = this.#deps.subscribe ?? DEFAULT_SUBSCRIBE;
    this.#sub = subscribe((e) => {
      if (isPackConfigEvent(e)) void this.load();
    });
  }

  stop(): void {
    this.#sub?.close();
    this.#sub = null;
  }

  async rollback(): Promise<boolean> {
    const ok = await this.active.rollback();
    if (ok) await this.load();
    return ok;
  }

  /** Deletes a stored pack at the revision the list served. */
  async remove(pack: PromptPackSummary): Promise<boolean> {
    if (this.busy || pack.revision == null) return false;
    this.busy = true;
    this.deleteError = null;
    try {
      await (this.#deps.deletePromptPack ?? deletePromptPack)(pack.name, pack.revision);
      await this.load();
      return true;
    } catch (e) {
      this.deleteError = packErrorText(e);
      if (packErrorDetail(e)?.error === 'revision_conflict') await this.load();
      return false;
    } finally {
      this.busy = false;
    }
  }

  clearClone(): void {
    this.cloneError = null;
    this.cloneReport = null;
  }

  /** Clones `from` into a new stored pack; the new doc, or null (the
   *  served refusal is in `cloneError` / `cloneReport`). */
  async clone(
    from: CloneSource,
    newName: string,
    description: string,
  ): Promise<PromptPackDoc | null> {
    if (this.busy) return null;
    this.busy = true;
    this.clearClone();
    try {
      return await (this.#deps.clonePromptPack ?? clonePromptPack)(from.name, {
        new_name: newName.trim(),
        revision: null,
        source: from.source,
        description: description.trim() || null,
      });
    } catch (e) {
      this.cloneError = packErrorText(e);
      this.cloneReport = packErrorDetail(e)?.report ?? null;
      return null;
    } finally {
      this.busy = false;
    }
  }
}

export function createPackList(deps: Partial<PackListDeps> = {}): PackList {
  return new PackList(deps);
}
