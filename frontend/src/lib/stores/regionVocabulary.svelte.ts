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
  type RegionTextRules,
  type RegionVocabularyEntry,
  type RegionVocabularyRole,
} from '$lib/api';

/** Titlecases a `snake_case` id as a display-label placeholder for a
 *  vocabulary the backend doesn't serve labels for yet (dq-region
 *  `text_choices`/`invalid_reasons` today; rejection reasons until
 *  openprocessor fix #29 lands `region_rejection_reason` labels on
 *  `GET {API_PREFIX}/regions/vocabulary`). Replace the call site with the
 *  served label the moment the backend adds one — this is a stand-in,
 *  not a hand-maintained label table. */
function titlecaseId(id: string): string {
  return id.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

class RegionVocabularyStore {
  detectors = $state<RegionVocabularyEntry[]>([]);
  regionSources = $state<RegionVocabularyEntry[]>([]);
  chainActors = $state<RegionVocabularyEntry[]>([]);
  /** `region_text_choice` values this deployment can serve. */
  textChoices = $state<string[]>([]);
  /** The active profile's region-text validity rules, or `null`. */
  textRules = $state<RegionTextRules | null>(null);
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
        this.textChoices = res.text_choices;
        this.textRules = res.text_rules;
      } catch {
        this.detectors = [];
        this.regionSources = [];
        this.chainActors = [];
        this.textChoices = [];
        this.textRules = null;
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

  /** Human label for a `region_text_choice` id — served `text_choices` is
   *  a plain id list today (no label), so this titlecases as a
   *  placeholder. */
  textChoiceLabel(id: string | null | undefined): string {
    if (!id) return '—';
    return titlecaseId(id);
  }

  /** Human label for a `region_text_vlm_invalid` reason id. */
  invalidReasonLabel(id: string | null | undefined): string {
    if (!id) return '—';
    return titlecaseId(id);
  }

  /**
   * Human label for a `region_rejection_reason` id (dq-region,
   * 2026-09-24) — `sanity_reject:<gate>`, `region_visible_elsewhere`,
   * `verifier_no_verdict`, or an older free-text reason. `openprocessor`
   * fix #29 will add labeled entries to `GET {API_PREFIX}/regions/vocabulary`
   * (with a flag distinguishing a model verdict from "needs human,
   * verdict inconclusive"); until then this renders the raw served id
   * titlecased, verbatim reason text unchanged — deliberately does NOT
   * infer "wrong box" from `verify_rejected` alone, since
   * `verifier_no_verdict` means the opposite (needs human review, not a
   * model rejection). That distinction is `region_bbox_correct`
   * (`false` = model said wrong box), not this reason id. */
  rejectionReasonLabel(id: string | null | undefined): string {
    if (!id) return '—';
    if (id.startsWith('sanity_reject:')) {
      return `Sanity check failed: ${titlecaseId(id.slice('sanity_reject:'.length))}`;
    }
    return titlecaseId(id);
  }
}

export const regionVocabularyStore = new RegionVocabularyStore();
