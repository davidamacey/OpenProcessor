<script lang="ts">
  import { getThumbUrl, getSourceImageWithBbox } from '$lib/api';
  import type { OpCrop, LabelSource } from '$lib/types';

  interface Props {
    crop: OpCrop;
    selected?: boolean;
    onclick?: (crop: OpCrop, e: MouseEvent) => void;
    onacceptGemma?: (crop: OpCrop) => void;
    onrejectGemma?: (crop: OpCrop) => void;
  }

  let {
    crop,
    selected = false,
    onclick,
    onacceptGemma,
    onrejectGemma,
  }: Props = $props();

  let expanded = $state<boolean>(false);

  const labelBadgeClass = (src: LabelSource | null | undefined, validated: boolean): string => {
    if (validated) return 'bg-green-500/20 text-green-200 border-green-500/40';
    if (src === 'gemma_suggestion') return 'bg-yellow-500/20 text-yellow-200 border-yellow-500/40';
    if (src === 'human_confirmed') return 'bg-green-500/20 text-green-200 border-green-500/40';
    return 'bg-blue-500/20 text-blue-200 border-blue-500/40';
  };

  const labelBadgeText = (src: LabelSource | null | undefined, validated: boolean): string => {
    if (validated) return 'human';
    if (src === 'gemma_suggestion') return 'gemma';
    if (src === 'model_suggestion') return 'model';
    if (src === 'cluster_propagation') return 'cluster';
    if (src === 'v6_original_label') return 'v6';
    return src ?? 'unknown';
  };

  const conf = $derived(
    crop.label_confidence != null ? `${(crop.label_confidence * 100).toFixed(0)}%` : null,
  );
</script>

<div
  role="button"
  tabindex="0"
  class="group relative flex flex-col rounded-md border bg-zinc-900 text-left transition focus:outline-none {selected
    ? 'border-blue-500 ring-2 ring-blue-500/40'
    : 'border-zinc-800 hover:border-zinc-600'}"
  onclick={(e) => onclick?.(crop, e)}
  onkeydown={(e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onclick?.(crop, e as unknown as MouseEvent);
    }
  }}
  aria-pressed={selected}
>
  <div class="relative aspect-square w-full overflow-hidden rounded-t-md bg-zinc-950">
    <img
      src={getThumbUrl(crop.id)}
      alt="crop {crop.id}"
      loading="lazy"
      class="h-full w-full object-contain"
      onerror={(e) => {
        const t = e.currentTarget as HTMLImageElement;
        t.style.opacity = '0.2';
      }}
    />

    {#if crop.plate_bbox_norm}
      <span
        class="absolute top-1 left-1 rounded-sm border border-blue-400/60 bg-blue-500/30 px-1 py-0.5 font-mono text-[10px] text-white"
      >
        plate
      </span>
    {/if}

    {#if conf}
      <span
        class="absolute right-1 bottom-1 rounded-sm bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-white"
      >
        {conf}
      </span>
    {/if}

    <button
      type="button"
      class="absolute top-1 right-1 rounded-sm bg-black/60 px-1.5 py-0.5 text-[10px] text-white opacity-0 transition group-hover:opacity-100"
      onclick={(e) => {
        e.stopPropagation();
        expanded = true;
      }}
      aria-label="Expand"
    >
      view
    </button>
  </div>

  <div class="flex items-center gap-1 px-2 py-1.5">
    <span
      class="truncate rounded-sm border px-1 py-0.5 text-[10px] font-medium {labelBadgeClass(
        crop.label_source,
        crop.label_validated,
      )}"
      title={crop.class_name ?? 'unlabeled'}
    >
      {labelBadgeText(crop.label_source, crop.label_validated)}
    </span>
    <span class="grow truncate text-xs text-zinc-300" title={crop.class_name ?? ''}>
      {crop.class_name ?? '—'}
    </span>
  </div>

  {#if crop.gemma_suggested_class_id != null && !crop.label_validated}
    <div
      class="flex items-center gap-1 border-t border-zinc-800 bg-yellow-500/5 px-2 py-1 text-xs"
    >
      <span class="grow truncate text-yellow-200" title={crop.gemma_suggested_class_name ?? ''}>
        Gemma: {crop.gemma_suggested_class_name ?? '—'}
        {#if crop.gemma_suggested_confidence != null}
          <span class="ml-1 text-yellow-400/70"
            >{(crop.gemma_suggested_confidence * 100).toFixed(0)}%</span
          >
        {/if}
      </span>
      <button
        type="button"
        class="rounded border border-green-500/40 bg-green-500/20 px-1 text-green-200 hover:bg-green-500/30"
        onclick={(e) => {
          e.stopPropagation();
          onacceptGemma?.(crop);
        }}
        aria-label="Accept Gemma suggestion"
      >
        ✓
      </button>
      <button
        type="button"
        class="rounded border border-red-500/40 bg-red-500/20 px-1 text-red-200 hover:bg-red-500/30"
        onclick={(e) => {
          e.stopPropagation();
          onrejectGemma?.(crop);
        }}
        aria-label="Reject Gemma suggestion"
      >
        ×
      </button>
    </div>
  {/if}
</div>

{#if expanded}
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Source image"
  >
    <div class="relative max-h-full max-w-6xl">
      <img
        src={getSourceImageWithBbox(crop.id)}
        alt="source"
        class="max-h-[85vh] max-w-full rounded-md border border-zinc-700"
      />
      <button
        type="button"
        class="absolute -top-3 -right-3 rounded-full border border-zinc-700 bg-zinc-900 px-2 py-1 text-sm text-white"
        onclick={() => (expanded = false)}
        aria-label="Close"
      >
        ×
      </button>
    </div>
  </div>
{/if}
