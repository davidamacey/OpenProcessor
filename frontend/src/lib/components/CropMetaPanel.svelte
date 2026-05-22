<script lang="ts">
  import type { OpCrop } from '$lib/types';
  import DetectorChip from './DetectorChip.svelte';

  interface Props {
    crop: OpCrop;
  }

  let { crop }: Props = $props();

  // Server emits a handful of fields that aren't yet in OpCrop's TS
  // shape (gemma_confidence categorical, plate_status string,
  // class_source string). Read them via narrow casts so this panel
  // doesn't need a public-type widening in the same PR.
  const extra = $derived(
    crop as unknown as {
      gemma_confidence?: string | null;
      plate_status?: string | null;
      class_source?: string | null;
    },
  );
  const gemmaConf = $derived<string | null>(extra.gemma_confidence ?? null);
  const plateStatus = $derived<string | null>(extra.plate_status ?? null);
  const classSource = $derived<string | null>(extra.class_source ?? null);

  const hasPlate = $derived(
    !!crop.plate_bbox_norm ||
      !!crop.plate_detector ||
      !!crop.plate_text ||
      plateStatus != null,
  );

  function pct(value: number | null | undefined): string {
    return value == null ? '—' : `${(value * 100).toFixed(1)}%`;
  }
</script>

<dl class="grid grid-cols-2 gap-y-1 text-xs">
  <dt class="text-zinc-500">Class</dt>
  <dd class="text-zinc-200">
    {crop.class_name ?? '—'}
    {#if classSource}
      <span class="ml-1 text-zinc-500">({classSource})</span>
    {/if}
  </dd>

  <dt class="text-zinc-500">Label source</dt>
  <dd class="text-zinc-200">
    {crop.label_source ?? '—'}
    {#if crop.label_validated}
      <span
        class="ml-1 rounded border border-green-500/40 bg-green-500/15 px-1 text-[10px] text-green-200"
      >
        validated
      </span>
    {/if}
  </dd>

  <dt class="text-zinc-500">Confidence</dt>
  <dd class="font-mono">
    {pct(crop.label_confidence)}
    {#if gemmaConf}
      <span class="ml-2 text-zinc-500">gemma:</span>
      <span class="text-zinc-200">{gemmaConf}</span>
    {/if}
  </dd>

  {#if crop.gemma_suggested_class_id != null}
    <dt class="text-zinc-500">Gemma proposed</dt>
    <dd class="text-yellow-200">
      {crop.gemma_suggested_class_name ?? '—'}
      {#if crop.gemma_suggested_confidence != null}
        <span class="ml-1 font-mono text-yellow-400/70">
          {pct(crop.gemma_suggested_confidence)}
        </span>
      {/if}
    </dd>
  {/if}

  <dt class="text-zinc-500">Cluster</dt>
  <dd class="font-mono text-zinc-200">
    #{crop.cluster_id ?? '—'}
    {#if crop.similarity_to_centroid != null}
      <span class="ml-1 text-zinc-500">sim {pct(crop.similarity_to_centroid)}</span>
    {/if}
  </dd>

  {#if crop.class_detector}
    <dt class="text-zinc-500">Class detector</dt>
    <dd>
      <DetectorChip detector={crop.class_detector} version={crop.class_detector_version} />
    </dd>
  {/if}

  {#if crop.class_labeled_at}
    <dt class="text-zinc-500">Labeled at</dt>
    <dd class="font-mono text-zinc-400">{crop.class_labeled_at}</dd>
  {/if}

  {#if crop.class_labeler}
    <dt class="text-zinc-500">Labeler</dt>
    <dd class="text-zinc-300">{crop.class_labeler}</dd>
  {/if}
</dl>

{#if hasPlate}
  <div class="mt-4 border-t border-zinc-800 pt-3">
    <div class="mb-1.5 text-[10px] uppercase tracking-wider text-zinc-500">Plate</div>
    <dl class="grid grid-cols-2 gap-y-1 text-xs">
      {#if plateStatus}
        <dt class="text-zinc-500">Status</dt>
        <dd class="text-zinc-200">{plateStatus}</dd>
      {/if}

      {#if crop.plate_score != null}
        <dt class="text-zinc-500">Score</dt>
        <dd class="font-mono">{pct(crop.plate_score)}</dd>
      {/if}

      {#if crop.plate_detector || crop.plate_verifier}
        <dt class="text-zinc-500">Detector</dt>
        <dd class="flex flex-wrap items-center gap-1.5">
          {#if crop.plate_detector}
            <DetectorChip
              detector={crop.plate_detector}
              version={crop.plate_detector_version}
            />
          {/if}
          {#if crop.plate_verifier}
            <DetectorChip
              detector={crop.plate_verifier}
              tag="verify"
              version={crop.plate_verifier_version}
              size="sm"
            />
          {/if}
        </dd>
      {/if}

      {#if crop.plate_detector_chain && crop.plate_detector_chain.length > 0}
        <dt class="text-zinc-500">Cascade</dt>
        <dd class="flex flex-wrap items-center gap-1">
          {#each crop.plate_detector_chain as entry (entry)}
            <DetectorChip raw={entry} size="sm" />
          {/each}
        </dd>
      {/if}

      {#if crop.plate_text != null}
        <dt class="text-zinc-500">Text</dt>
        <dd class="flex items-center gap-1.5">
          <span class="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 font-mono text-zinc-100">
            {crop.plate_text || '∅'}
          </span>
          {#if crop.plate_text_source}
            <DetectorChip detector={crop.plate_text_source} size="sm" />
          {/if}
          {#if crop.plate_text_confidence != null}
            <span class="font-mono text-[10px] text-zinc-500">
              {pct(crop.plate_text_confidence)}
            </span>
          {/if}
        </dd>
      {/if}

      {#if crop.plate_rejection_reason}
        <dt class="text-zinc-500">Rejection</dt>
        <dd class="text-zinc-300">{crop.plate_rejection_reason}</dd>
      {/if}
    </dl>
  </div>
{/if}

{#if crop.source_image_path || crop.updated_at}
  <div class="mt-4 border-t border-zinc-800 pt-3 text-[11px] text-zinc-500">
    {#if crop.source_image_path}
      <div class="truncate" title={crop.source_image_path}>
        <span class="text-zinc-600">src:</span> {crop.source_image_path}
      </div>
    {/if}
    {#if crop.updated_at}
      <div>
        <span class="text-zinc-600">updated:</span>
        <span class="font-mono">{crop.updated_at}</span>
      </div>
    {/if}
    <div>
      <span class="text-zinc-600">id:</span>
      <span class="font-mono">{crop.id}</span>
    </div>
  </div>
{/if}
