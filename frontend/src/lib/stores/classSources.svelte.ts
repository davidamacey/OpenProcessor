/**
 * ClassSourcesStore — the deployment's `class_source` catalog
 * (`GET {API_PREFIX}/class_sources`), loaded once. The ingest detectors'
 * values are named from each deployment's config, so this is the only
 * way the UI knows them. On failure the catalog is empty and callers
 * render raw ids verbatim (the backend's documented fallback for ids
 * outside the catalog).
 */

import { getClassSources, type ClassSource, type ClassSourceRole } from '$lib/api';

class ClassSourcesStore {
  list = $state<ClassSource[]>([]);
  loaded = $state<boolean>(false);
  #byId = $derived(new Map(this.list.map((c) => [c.id, c])));
  #inflight: Promise<void> | null = null;

  async init(): Promise<void> {
    if (this.loaded) return;
    if (this.#inflight) return this.#inflight;
    this.#inflight = (async () => {
      try {
        this.list = await getClassSources();
      } catch {
        this.list = [];
      } finally {
        this.loaded = true;
        this.#inflight = null;
      }
    })();
    return this.#inflight;
  }

  /** Human label for a `class_source` id; the id itself when unknown. */
  labelFor(id: string | null | undefined): string {
    if (!id) return '';
    return this.#byId.get(id)?.label ?? id;
  }

  roleFor(id: string | null | undefined): ClassSourceRole | null {
    if (!id) return null;
    return this.#byId.get(id)?.role ?? null;
  }
}

export const classSourcesStore = new ClassSourcesStore();
