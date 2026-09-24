/**
 * RegionVocabularyStore — the deployment-configured detector/segmenter/
 * verifier vocabulary (`GET {API_PREFIX}/regions/vocabulary`, W0 naming-sweep
 * finding m9), loaded once. Built by the backend from the active region
 * profile / ingest profiles / `OP_VLM_MODEL` — never a hardcoded model
 * id — so this is the only way the UI knows a detector's human label and
 * role. `ProvenanceChip.svelte` reads the label from here and derives
 * chip color from the role (`detectorRegistry.ts`'s `paletteForRole`);
 * `SlotGallery.svelte`'s detector filter reads `filterableDetectors`.
 *
 * On failure (or before `init()` resolves) every list is empty and
 * `labelFor`/`roleFor` fall back to rendering the raw id verbatim with a
 * neutral chip — the documented "unknown id" behavior — so a missing
 * endpoint never breaks a page that renders provenance chips.
 */

import {
  getRegionVocabulary,
  type RegionVocabularyEntry,
  type RegionVocabularyRole,
} from '$lib/api';

class RegionVocabularyStore {
  detectors = $state<RegionVocabularyEntry[]>([]);
  regionSources = $state<RegionVocabularyEntry[]>([]);
  chainActors = $state<RegionVocabularyEntry[]>([]);
  loaded = $state<boolean>(false);
  #inflight: Promise<void> | null = null;

  // Chip-rendering call sites resolve an id against any of the three
  // vocabularies at once — a chain entry's head can be a detector, a
  // chain actor (e.g. the VLM verifier), or (rarely) a region source —
  // so lookups fold all three into one map rather than making every
  // caller know which list to check first.
  #byId = $derived(
    new Map(
      [...this.detectors, ...this.chainActors, ...this.regionSources].map((e) => [
        e.id,
        e,
      ]),
    ),
  );

  filterableDetectors = $derived(this.detectors.filter((d) => d.filterable));

  async init(): Promise<void> {
    if (this.loaded) return;
    if (this.#inflight) return this.#inflight;
    this.#inflight = (async () => {
      try {
        const res = await getRegionVocabulary();
        this.detectors = res.detectors;
        this.regionSources = res.region_sources;
        this.chainActors = res.chain_actors;
      } catch {
        this.detectors = [];
        this.regionSources = [];
        this.chainActors = [];
      } finally {
        this.loaded = true;
        this.#inflight = null;
      }
    })();
    return this.#inflight;
  }

  /** Human label for a detector/chain-actor/region-source id; the id
   *  itself when unknown or absent (matches ProvenanceChip's documented
   *  "unknown id renders verbatim" contract). */
  labelFor(id: string | null | undefined): string {
    if (!id) return '—';
    return this.#byId.get(id)?.label ?? id;
  }

  roleFor(id: string | null | undefined): RegionVocabularyRole | null {
    if (!id) return null;
    return this.#byId.get(id)?.role ?? null;
  }
}

export const regionVocabularyStore = new RegionVocabularyStore();
