<script lang="ts">
  /**
   * Provenance chip for ML detector outputs.
   *
   * Used on the /review plates panel and the /clusters plates list to
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
   * Label/palette resolution is config-driven (P2.2,
   * docs/genericization-plan-2026-09-13.md §3.2): this component used
   * to hand-code a 20-arm label switch and a 10-branch palette
   * if-chain; both are now data (`legacyDetectorRegistry`), proven
   * equivalent to the old functions by a 23-case snapshot test
   * (`annotations/legacyDetectors.test.ts`) before they were deleted.
   */
  import {
    labelForDetector,
    paletteForDetector,
    isMutedTag as isMutedTagFor,
  } from '$lib/annotations/detectorRegistry';
  import { legacyDetectorRegistry } from '$lib/annotations/profiles/legacyDetectors';

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

  const palette = $derived(paletteForDetector(legacyDetectorRegistry, parsed.detector));
  const muted = $derived(isMutedTagFor(legacyDetectorRegistry, parsed.tag));
  const sizeCls = $derived(size === 'sm' ? 'px-1.5 text-[10px]' : '');
</script>

<span
  class="chip {palette.border} {palette.bg} {palette.text} {sizeCls} font-mono {muted
    ? 'opacity-60'
    : ''}"
  title={version ? `${parsed.detector ?? ''} v${version}` : (parsed.detector ?? '')}
>
  <span>{labelForDetector(legacyDetectorRegistry, parsed.detector)}</span>
  {#if parsed.tag}
    <span class="text-[9px] uppercase tracking-wide opacity-80">{parsed.tag}</span>
  {/if}
</span>
