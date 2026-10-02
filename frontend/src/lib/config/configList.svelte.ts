/**
 * A config resource's list page state, shared by the prompt-pack (W3) and
 * region-profile (W4) lists (any_domain_plan.md §7.2, §7.3, §7.6 items 1,
 * 2 and 4): the served list, the active ref (with its writes), Clone and
 * Delete.
 *
 * A `config.changed` event on the resource's axis is a wake-up to re-read;
 * the served list stays the only source.
 */
import { configErrorDetail, configErrorText } from '$lib/api';
import type { CurationEvent } from '$lib/sse';
import type {
  ConfigCloneRequest,
  ConfigSource,
  ValidationReport,
} from '$lib/types_config';
import type { ConfigActive } from './configActive.svelte';

export interface ConfigListBackend<L, D> {
  list: () => Promise<L>;
  clone: (name: string, body: ConfigCloneRequest) => Promise<D>;
  remove: (name: string, expectedRevision: number) => Promise<void>;
  subscribe: (onEvent: (e: CurationEvent) => void) => { close(): void };
  isEvent: (e: CurationEvent) => boolean;
}

/** What the clone dialog clones: a doc, or a template (`source`). */
export interface CloneSource {
  name: string;
  source: ConfigSource | null;
}

export class ConfigList<L, D, A extends ConfigActive = ConfigActive> {
  list = $state<L | null>(null);
  loadError = $state<string | null>(null);
  readonly active: A;

  busy = $state(false);
  deleteError = $state<string | null>(null);
  cloneError = $state<string | null>(null);
  cloneReport = $state<ValidationReport | null>(null);

  protected backend: ConfigListBackend<L, D>;
  #sub: { close(): void } | null = null;

  constructor(backend: ConfigListBackend<L, D>, active: A) {
    this.backend = backend;
    this.active = active;
  }

  async load(): Promise<void> {
    await Promise.all([
      (async () => {
        try {
          this.list = await this.backend.list();
          this.loadError = null;
        } catch (e) {
          if ((e as Error)?.name === 'AbortError') return;
          this.loadError = configErrorText(e);
        }
      })(),
      this.active.load(),
    ]);
  }

  start(): void {
    void this.load();
    this.#sub = this.backend.subscribe((e) => {
      if (this.backend.isEvent(e)) void this.load();
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

  /** Deletes a stored doc at the revision the list served. */
  async remove(row: { name: string; revision: number | null }): Promise<boolean> {
    if (this.busy || row.revision == null) return false;
    this.busy = true;
    this.deleteError = null;
    try {
      await this.backend.remove(row.name, row.revision);
      await this.load();
      return true;
    } catch (e) {
      this.deleteError = configErrorText(e);
      if (configErrorDetail(e)?.error === 'revision_conflict') await this.load();
      return false;
    } finally {
      this.busy = false;
    }
  }

  clearClone(): void {
    this.cloneError = null;
    this.cloneReport = null;
  }

  /** Clones `from` into a new stored doc; the new doc, or null (the
   *  served refusal is in `cloneError` / `cloneReport`). */
  async clone(
    from: CloneSource,
    newName: string,
    description: string,
  ): Promise<D | null> {
    if (this.busy) return null;
    this.busy = true;
    this.clearClone();
    try {
      return await this.backend.clone(from.name, {
        new_name: newName.trim(),
        revision: null,
        source: from.source,
        description: description.trim() || null,
      });
    } catch (e) {
      this.cloneError = configErrorText(e);
      this.cloneReport = configErrorDetail(e)?.report ?? null;
      return null;
    } finally {
      this.busy = false;
    }
  }
}
