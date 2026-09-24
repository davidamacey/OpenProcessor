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
  type RejectionReasonEntry,
  type RejectionReasonKind,
} from '$lib/api';

/** Titlecases a `snake_case` id as a display-label placeholder for a
 *  vocabulary the backend doesn't serve labels for yet (dq-region
 *  `text_choices`/`invalid_reasons` — `region_rejection_reason` got its
 *  own labeled vocabulary in openprocessor fix #29, see
 *  `rejectionReasonLabel` below, so this no longer covers it). Replace
 *  the call site with the served label the moment the backend adds one
 *  — this is a stand-in, not a hand-maintained label table. */
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
  /** Labeled `region_rejection_reason` vocabulary (openprocessor fix #29,
   *  840beb8 adoption). */
  rejectionReasons = $state<RejectionReasonEntry[]>([]);
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
        this.rejectionReasons = res.rejection_reasons;
      } catch {
        this.detectors = [];
        this.regionSources = [];
        this.chainActors = [];
        this.textChoices = [];
        this.textRules = null;
        this.rejectionReasons = [];
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

  /** Resolves a stored `region_rejection_reason` value against the
   *  served vocabulary — an exact match first, then the longest matching
   *  `match: 'prefix'` entry (so `sanity_reject:<gate>` resolves through
   *  its `sanity_reject:` entry), `null` when nothing matches (an older
   *  free-text human reason). */
  #resolveRejectionReason(id: string): RejectionReasonEntry | null {
    const exact = this.rejectionReasons.find((e) => e.match === 'exact' && e.id === id);
    if (exact) return exact;
    const prefixMatches = this.rejectionReasons.filter(
      (e) => e.match === 'prefix' && id.startsWith(e.id),
    );
    if (prefixMatches.length === 0) return null;
    // Longest prefix wins in the (currently hypothetical) case of two
    // overlapping prefix entries.
    return prefixMatches.reduce((best, e) => (e.id.length > best.id.length ? e : best));
  }

  /**
   * Human label for a `region_rejection_reason` id (dq-region 2026-09-24,
   * labeled by openprocessor fix #29 / 840beb8) — `sanity_reject:<gate>`,
   * `region_visible_elsewhere`, `verifier_no_verdict`, or an older
   * free-text reason. Resolves an exact match first, then a prefix match
   * (`label_template`'s `{detail}` filled from whatever follows the
   * matched prefix), and falls back to the raw stored value verbatim —
   * never titlecased — when nothing in the served vocabulary matches.
   * Deliberately does NOT infer "wrong box" from `verify_rejected` alone,
   * since `verifier_no_verdict` means the opposite (needs human review,
   * not a model rejection) — see `rejectionReasonKind` and
   * `region_bbox_correct` (`false` = model said wrong box) for that
   * distinction. */
  rejectionReasonLabel(id: string | null | undefined): string {
    if (!id) return '—';
    const entry = this.#resolveRejectionReason(id);
    if (!entry) return id;
    if (entry.match === 'prefix' && entry.label_template) {
      const detail = id.slice(entry.id.length);
      return entry.label_template.replace('{detail}', detail);
    }
    return entry.label;
  }

  /** The served `kind` for a `region_rejection_reason` id — `null` when
   *  the value isn't in the served vocabulary (an older free-text human
   *  reason). Drives styling: `model_verdict` (the verifier judged the
   *  box wrong) vs `automatic` (a geometry gate) vs `needs_human` (no
   *  verdict at all — must never be worded as a rejection). */
  rejectionReasonKind(id: string | null | undefined): RejectionReasonKind | null {
    if (!id) return null;
    return this.#resolveRejectionReason(id)?.kind ?? null;
  }
}

export const regionVocabularyStore = new RegionVocabularyStore();
