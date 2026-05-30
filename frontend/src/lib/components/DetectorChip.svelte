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
   *   <DetectorChip detector="lpr_nanov11_640" />
   *     → blue "LPR" chip
   *
   *   <DetectorChip raw="lpr_nanov11_640:miss" />
   *     → blue "LPR miss" chip with muted opacity
   *
   * The `raw` form parses entries from `plate_detector_chain` so the
   * meta panel can render the full cascade story as a chip strip.
   */

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

  // Short, readable label per detector. Anything unknown falls back to
  // the raw string so the UI never silently swallows a new detector
  // the backend started emitting.
  function labelFor(d: string | null): string {
    if (!d) return '—';
    switch (d) {
      case 'lpr_nanov11_640': return 'LPR';
      case 'sam3': return 'SAM3';
      case 'paddleocr_det_trt': return 'Paddle det';
      case 'paddleocr_rec_trt': return 'Paddle rec';
      case 'paddleocr_rec': return 'Paddle rec';
      case 'paddleocr_det': return 'Paddle det';
      case 'paddleocr': return 'Paddle';
      case 'human': return 'Human';
      case 'gemma-4-e4b': return 'Gemma';
      case 'gemma': return 'Gemma';
      case 'gemma_propose': return 'Gemma';
      case 'gemma_prefilter': return 'Gemma prefilter';
      case 'legacy_vehicle_v6_trt': return 'v6';
      case 'yolov11_small_trt_end2end': return 'YOLO11';
      case 'coco_yolo11_proposal': return 'YOLO11';
      case 'ingest_v6': return 'Ingest v6';
      // Quantization-variant runtimes (bake-off / quantization panel).
      case 'onnxruntime': return 'ORT';
      case 'ort-cuda': return 'ORT·CUDA';
      case 'ort-trt': return 'ORT·TRT';
      case 'ort-cpu': return 'ORT·CPU';
      case 'coreml': return 'CoreML';
      default: return d;
    }
  }

  // Color family by detector. Miss/reject tags get a muted background.
  function paletteFor(d: string | null): { border: string; bg: string; text: string } {
    if (!d) return { border: 'border-zinc-700', bg: 'bg-zinc-800/60', text: 'text-zinc-300' };
    if (d.startsWith('lpr')) {
      return { border: 'border-blue-500/50', bg: 'bg-blue-500/10', text: 'text-blue-200' };
    }
    if (d.startsWith('sam')) {
      return { border: 'border-purple-500/50', bg: 'bg-purple-500/10', text: 'text-purple-200' };
    }
    if (d.startsWith('paddle')) {
      return { border: 'border-amber-500/50', bg: 'bg-amber-500/10', text: 'text-amber-200' };
    }
    if (d === 'human') {
      return { border: 'border-emerald-500/50', bg: 'bg-emerald-500/10', text: 'text-emerald-200' };
    }
    if (d.startsWith('gemma')) {
      return { border: 'border-teal-500/50', bg: 'bg-teal-500/10', text: 'text-teal-200' };
    }
    if (d === 'legacy_vehicle_v6_trt' || d === 'ingest_v6') {
      return { border: 'border-rose-500/50', bg: 'bg-rose-500/10', text: 'text-rose-200' };
    }
    if (d.startsWith('yolov11') || d === 'coco_yolo11_proposal') {
      return { border: 'border-sky-500/50', bg: 'bg-sky-500/10', text: 'text-sky-200' };
    }
    if (d.startsWith('ort') || d === 'onnxruntime') {
      return { border: 'border-indigo-500/50', bg: 'bg-indigo-500/10', text: 'text-indigo-200' };
    }
    if (d.startsWith('coreml')) {
      return { border: 'border-orange-500/50', bg: 'bg-orange-500/10', text: 'text-orange-200' };
    }
    return { border: 'border-zinc-700', bg: 'bg-zinc-800/60', text: 'text-zinc-300' };
  }

  // Muted variant for miss/reject tags so the operator's eye is drawn to
  // the winning detector and away from the cascade attempts that failed.
  function isMutedTag(t: string | null): boolean {
    if (!t) return false;
    return /miss|reject|skipped|degenerate|unparseable/.test(t);
  }

  const palette = $derived(paletteFor(parsed.detector));
  const muted = $derived(isMutedTag(parsed.tag));
  const sizeCls = $derived(
    size === 'sm' ? 'px-1.5 py-0.5 text-[10px]' : 'px-2 py-0.5 text-[11px]',
  );
</script>

<span
  class="inline-flex items-center gap-1 rounded border {palette.border} {palette.bg} {palette.text} {sizeCls} font-mono {muted
    ? 'opacity-60'
    : ''}"
  title={version ? `${parsed.detector ?? ''} v${version}` : parsed.detector ?? ''}
>
  <span>{labelFor(parsed.detector)}</span>
  {#if parsed.tag}
    <span class="text-[9px] uppercase tracking-wide opacity-80">{parsed.tag}</span>
  {/if}
</span>
