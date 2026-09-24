<script lang="ts">
  import type { OpCrop } from '$lib/types';
  import ProvenanceChip from './ProvenanceChip.svelte';
  import { slotRegistry } from '$lib/annotations/registeredSlots';
  import { slotOf } from '$lib/annotations/cropSlots';
  import { slotIsPresent } from '$lib/annotations/types';

  interface Props {
    crop: OpCrop;
  }

  let { crop }: Props = $props();

  const vlmConf = $derived<string | null>(crop.vlm_confidence ?? null);
  const classSource = $derived<string | null>(crop.class_source ?? null);

  // Slot(s) bound to this crop's class. A one-entry map built from the
  // crop's own class_id/class_name is all forClass() needs — this panel
  // has no other source of a classesById lookup, and every crop already
  // carries the one class name that matters for its own binding check.
  const boundSlots = $derived(
    crop.class_id != null
      ? slotRegistry.forClass(
          crop.class_id,
          new Map([[crop.class_id, crop.class_name ?? '']]),
        )
      : [],
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
    {#if vlmConf}
      <span class="ml-2 text-zinc-500">VLM:</span>
      <span class="text-zinc-200">{vlmConf}</span>
    {/if}
  </dd>

  {#if crop.vlm_suggested_class_id != null}
    <dt class="text-zinc-500">VLM proposed</dt>
    <dd class="text-yellow-200">
      {crop.vlm_suggested_class_name ?? '—'}
      {#if crop.vlm_confidence}
        <span class="ml-1 font-mono text-yellow-400/70">{crop.vlm_confidence}</span>
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
      <ProvenanceChip
        detector={crop.class_detector}
        version={crop.class_detector_version}
      />
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

{#each boundSlots as spec (spec.key)}
  {@const data = slotOf(crop, spec)}
  {#if slotIsPresent(data)}
    <div class="mt-4 border-t border-zinc-800 pt-3">
      <div class="mb-1.5 text-[10px] uppercase tracking-wider text-zinc-500">
        {spec.label.title}
      </div>
      <dl class="grid grid-cols-2 gap-y-1 text-xs">
        {#if data?.lifecycle?.state}
          <dt class="text-zinc-500">Status</dt>
          <dd class="text-zinc-200">{data.lifecycle.state.label}</dd>
        {:else if data?.lifecycle?.status}
          <dt class="text-zinc-500">Status</dt>
          <dd class="text-zinc-200">{data.lifecycle.status}</dd>
        {/if}

        {#if data?.subBox?.score != null}
          <dt class="text-zinc-500">Score</dt>
          <dd class="font-mono">{pct(data.subBox.score)}</dd>
        {/if}

        {#if data?.provenance?.detector || data?.provenance?.verifier}
          <dt class="text-zinc-500">Detector</dt>
          <dd class="flex flex-wrap items-center gap-1.5">
            {#if data.provenance.detector}
              <ProvenanceChip
                detector={data.provenance.detector}
                version={data.provenance.detectorVersion}
              />
            {/if}
            {#if data.provenance.verifier}
              <ProvenanceChip
                detector={data.provenance.verifier}
                tag="verify"
                version={data.provenance.verifierVersion}
                size="sm"
              />
            {/if}
          </dd>
        {/if}

        {#if data?.provenance?.chain && data.provenance.chain.length > 0}
          <dt class="text-zinc-500">Cascade</dt>
          <dd class="flex flex-wrap items-center gap-1">
            {#each data.provenance.chain as entry (entry)}
              <ProvenanceChip raw={entry} size="sm" />
            {/each}
          </dd>
        {/if}

        {#if data?.text?.value != null}
          <dt class="text-zinc-500">Text</dt>
          <dd class="flex items-center gap-1.5">
            <span
              class="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 font-mono text-zinc-100"
            >
              {data.text.value || '∅'}
            </span>
            {#if data.text.source}
              <ProvenanceChip detector={data.text.source} size="sm" />
            {/if}
            {#if data.text.confidence != null}
              <span class="font-mono text-[10px] text-zinc-500">
                {pct(data.text.confidence)}
              </span>
            {/if}
          </dd>
        {/if}

        {#if data?.lifecycle?.rejectionReason}
          <dt class="text-zinc-500">Rejection</dt>
          <dd class="text-zinc-300">{data.lifecycle.rejectionReason}</dd>
        {/if}
      </dl>
    </div>
  {/if}
{/each}

{#if crop.source_image_path || crop.updated_at}
  <div class="mt-4 border-t border-zinc-800 pt-3 text-[11px] text-zinc-500">
    {#if crop.source_image_path}
      <div class="truncate" title={crop.source_image_path}>
        <span class="text-zinc-600">src:</span>
        {crop.source_image_path}
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
