<script lang="ts">
  /**
   * Provenance chip for ML detector outputs.
   *
   * Used on the /review slot panel and the /clusters slot gallery to
   * tell an operator at a glance whether a bbox came from LPR, SAM3,
   * PaddleOCR, Gemma, or a human — and whether a particular step in
   * the cascade hit, missed, was rejected by Gemma, etc.
   *
   * Two forms:
   *
   *   <ProvenanceChip detector="lpr_nanov11_640" />
   *     → blue "LPR" chip
   *
   *   <ProvenanceChip raw="lpr_nanov11_640:miss" />
   *     → blue "LPR miss" chip with muted opacity
   *
   * The `raw` form parses entries from `region_detector_chain` so the
   * meta panel can render the full cascade story as a chip strip.
   *
   * Label/palette resolution is now backend-driven (W0 naming-sweep
   * finding m9): this component used to hand-code a 20-arm label switch
   * and a 10-branch palette if-chain, then (P2.2) moved that same table
   * into a config object. Both are gone — the label comes from
   * `GET {API_PREFIX}/regions/vocabulary` (`regionVocabularyStore`), and the chip
   * COLOR comes from that vocabulary entry's `role` via
   * `paletteForRole` (display-only, still local). An id the vocabulary
   * doesn't know about renders verbatim with the neutral chip.
   */
  import {
    paletteForRole,
    isMutedTag as isMutedTagFor,
  } from '$lib/annotations/detectorRegistry';
  import { builtinDetectorRegistry } from '$lib/annotations/profiles/builtinDetectors';
  import { regionVocabularyStore } from '$stores/regionVocabulary.svelte';

  interface Props {
    /** The detector that produced the stored bbox. Maps to a color family. */
    detector?: string | null;
    /** Optional outcome suffix — 'hit' / 'miss' / 'gemma_ok' / 'gemma_reject' / etc. */
    tag?: string | null;
    /** Convenience for chain entries like 'lpr_nanov11_640:miss'. Parses to detector+tag. */
    raw?: string | null;
    /** Model version, shown as a small subtitle when set. */
    version?: string | null;
    size?: 'sm' | 'md';
  }

  let { detector, tag, raw, version, size = 'md' }: Props = $props();

  // Allow `raw="<detector>:<tag>"` as a shorthand. Tags after the first
  // colon are joined back so multi-part tags ('box_prompt:hit') survive.
  const parsed = $derived.by(() => {
    if (raw) {
      const idx = raw.indexOf(':');
      if (idx >= 0) {
        return { detector: raw.slice(0, idx), tag: raw.slice(idx + 1) };
      }
      return { detector: raw, tag: null };
    }
    return { detector: detector ?? null, tag: tag ?? null };
  });

  const role = $derived(regionVocabularyStore.roleFor(parsed.detector));
  const palette = $derived(paletteForRole(role));
  const muted = $derived(isMutedTagFor(builtinDetectorRegistry, parsed.tag));
  const sizeCls = $derived(size === 'sm' ? 'px-1.5 text-[10px]' : '');
</script>

<!-- DQ-p2 (docs/design/data-quality-pass-2026-09-24.md): the raw form
     (`raw="combined_verify_reject:region_visible_elsewhere"`, from
     `region_detector_chain`) renders whatever the vocabulary doesn't
     recognize verbatim — a long snake_case actor/tag id, sized against
     the shared `.chip` class's `white-space: nowrap` with no cap, so it
     overflowed the card edge ("COMBIN…" clipped by the card's own
     `overflow-hidden`, not a graceful in-chip ellipsis). `max-w-*
     truncate` on each inner span caps and ellipsizes long text WITHIN
     the chip's own border instead — the full string is still available
     via `title` on hover. Scoped to these two inner spans rather than
     the shared `.chip` class itself, which many short, never-overflowing
     chips elsewhere (StrategyBar, filter chips) also use. -->
<span
  class="chip {palette.border} {palette.bg} {palette.text} {sizeCls} font-mono {muted
    ? 'opacity-60'
    : ''}"
  title={version ? `${parsed.detector ?? ''} v${version}` : (parsed.detector ?? '')}
>
  <span class="max-w-[140px] truncate"
    >{regionVocabularyStore.labelFor(parsed.detector)}</span
  >
  {#if parsed.tag}
    <span class="max-w-[140px] truncate text-[9px] uppercase tracking-wide opacity-80">
      {parsed.tag}
    </span>
  {/if}
</span>
