<script lang="ts">
  import type { Crop } from '$lib/types';
  import ProvenanceChip from './ProvenanceChip.svelte';
  import ConfirmationChip from './labelConfirmation/ConfirmationChip.svelte';
  import DetectorClass from './labelConfirmation/DetectorClass.svelte';
  import VlmProvenanceRows from './provenance/VlmProvenanceRows.svelte';
  import ImportProvenanceRows from './provenance/ImportProvenanceRows.svelte';
  import CombineOriginRows from './provenance/CombineOriginRows.svelte';
  import OpenVocabProvenanceRows from './provenance/OpenVocabProvenanceRows.svelte';
  import EmbeddingRows from './provenance/EmbeddingRows.svelte';
  import ReprocessControl from './datasets/ReprocessControl.svelte';
  import AssistSuggestions from './assistRun/AssistSuggestions.svelte';
  import CropMetaSlots from '$components/cropMeta/CropMetaSlots.svelte';
  import CropItemText from '$components/cropMeta/CropItemText.svelte';
  import CropHistoryBlock from '$components/cropMeta/CropHistoryBlock.svelte';
  import CropSourceContext from '$components/cropMeta/CropSourceContext.svelte';
  import { createCropMetaLoader } from '$lib/cropMeta/cropMetaLoader.svelte';
  import { pct } from '$lib/cropMeta/pct';
  import { classSourcesStore } from '$stores/classSources.svelte';
  import { humanizeId } from '$lib/humanizeId';
  import { formatTimestamp } from '$lib/format/date';

  interface Props {
    crop: Crop;
    /** Embedded under /review's own item panel, which already shows the
     *  class, label source, detector score and VLM confidence directly
     *  above — hide those rows here instead of repeating them (visual
     *  audit 2026-09-24, R11). */
    embedded?: boolean;
    /** The served post-write crop after a Reprocess, for the host to
     *  adopt in place of `crop`. */
    onreprocessed?: (crop: Crop) => void;
  }

  let { crop, embedded = false, onreprocessed }: Props = $props();

  const vlmConf = $derived<string | null>(crop.vlm_confidence ?? null);
  const classSource = $derived<string | null>(crop.class_source ?? null);
  // DQ-M8: served role, not a hardcoded string match against class_source
  // — the catalog (GET {API_PREFIX}/class_sources, classSourcesStore) is
  // the deployment's actual vocabulary (sourceBadge.ts uses the same
  // role.startsWith('vlm') check for the label-source badge).
  const isVlmSourced = $derived(
    (classSourcesStore.roleFor(classSource) ?? '').startsWith('vlm'),
  );

  // Served verbatim, never inferred client-side: 'vlm' | 'model' | null.
  const classConfidenceSourceLabel = $derived<string>(
    crop.class_confidence_source === 'vlm'
      ? 'VLM'
      : crop.class_confidence_source === 'model'
        ? 'Model'
        : '',
  );

  // History and source image context: lazy, fetched once per crop.id,
  // never blocking the rest of the panel from rendering.
  const loader = createCropMetaLoader(() => crop.id);
</script>

{#if crop.class_excluded}
  <div
    class="mb-3 rounded border border-amber-500/40 bg-amber-500/10 px-2 py-1.5 text-xs text-amber-200"
  >
    <span class="font-medium">Ignored</span>
    {#if crop.excluded_reason}
      <span class="ml-1 text-amber-300/80">({crop.excluded_reason})</span>
    {/if}
    {#if crop.excluded_at}
      <span class="ml-1 font-mono text-amber-300/60" title={crop.excluded_at}
        >{formatTimestamp(crop.excluded_at)}</span
      >
    {/if}
  </div>
{/if}

<dl class="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-xs">
  {#if !embedded}
    <dt class="text-zinc-500">Class</dt>
    <dd class="text-zinc-200">
      <!-- R6 (visual audit 2026-09-24): a class-less crop read
         "Class (vlm_new_class_pending)" with a blank name and a raw id. -->
      {#if crop.class_name}
        {crop.class_name}
      {:else}
        <span class="italic text-zinc-500">no class yet</span>
      {/if}
      {#if classSource}
        <span class="ml-1 text-zinc-500" title={classSource}
          >({classSourcesStore.labelFor(classSource)})</span
        >
      {/if}
    </dd>

    <dt class="text-zinc-500">Label source</dt>
    <dd class="text-zinc-200">
      {crop.label_source ?? '—'}
      <ConfirmationChip source={crop.class_source} validated={crop.class_validated} />
      {#if crop.class_validated}
        <span
          class="ml-1 rounded border border-green-500/40 bg-green-500/15 px-1 text-[10px] text-green-200"
        >
          validated
        </span>
      {/if}
    </dd>

    <!-- DQ-M8 (docs/design/data-quality-pass-2026-09-24.md): `label_confidence`
       (wire `confidence`) is the classifier-detector score on EVERY row,
       including ones the VLM labeled — never the VLM's own confidence.
       Calling it plain "Confidence" next to a VLM-sourced label reads as
       the VLM's certainty (the design doc's repro: "Confidence 94.6%"
       under "Current label widget_a (vlm)"). Label it for what it is
       whenever the class came from the VLM, and show the VLM's own
       categorical confidence (`vlm_confidence`, served separately) as
       its own row instead of folding it in as a same-row detail. -->
    <dt class="text-zinc-500">{isVlmSourced ? 'Detector score' : 'Confidence'}</dt>
    <dd class="font-mono">{pct(crop.label_confidence)}</dd>

    {#if isVlmSourced && crop.detector_class_name}
      <dt class="text-zinc-500">Detector class</dt>
      <dd class="text-zinc-200"><DetectorClass {crop} bare /></dd>
    {/if}

    {#if vlmConf}
      <dt class="text-zinc-500">VLM confidence</dt>
      <dd class="font-mono text-zinc-200">{vlmConf}</dd>
    {/if}
  {/if}

  <!-- Per-feature provenance rows (VLM, import/lock, open
       vocabulary, combine origin, embedding state): each
       component renders its own <dt>/<dd> pairs or nothing. -->
  <VlmProvenanceRows {crop} />
  <ImportProvenanceRows {crop} />
  <OpenVocabProvenanceRows {crop} />
  <CombineOriginRows {crop} />
  <EmbeddingRows {crop} />

  <!-- dq-queues cutover (2026-09-24): `class_confidence` is the served
       confidence of whoever set the LABEL (VLM categorical mapped to a
       number, or the classifier's own score) — distinct from
       `label_confidence` above, which is always the detector score. Null
       for human labels, so this row is omitted then. -->
  {#if crop.class_confidence != null}
    <dt class="text-zinc-500">Label confidence</dt>
    <dd class="font-mono text-zinc-200">
      {classConfidenceSourceLabel}
      {pct(crop.class_confidence)}
    </dd>
  {/if}

  {#if crop.vlm_raw_class}
    <dt class="text-zinc-500">VLM said</dt>
    <dd class="text-zinc-300">{crop.vlm_raw_class}</dd>
  {/if}

  {#if crop.assist}
    <dt class="text-zinc-500">Assist</dt>
    <dd data-testid="meta-assist">
      <AssistSuggestions assist={crop.assist} className={crop.class_name} />
    </dd>
  {/if}

  {#if crop.vlm_class_empty_reason}
    <dt class="text-zinc-500">VLM empty reason</dt>
    <dd class="text-orange-300" title={crop.vlm_class_empty_reason}>
      {humanizeId(crop.vlm_class_empty_reason)}
    </dd>
  {/if}

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
    <!-- p6 (2026-09-24 interactive pass): the ISO timestamp has no
         whitespace to wrap on, so it overflowed the fixed-width panel and
         got visually clipped ("…+00:0") instead of wrapping. -->
    <dd class="font-mono text-zinc-400" title={crop.class_labeled_at}>
      {formatTimestamp(crop.class_labeled_at)}
    </dd>
  {/if}

  {#if crop.class_labeler}
    <dt class="text-zinc-500">Labeler</dt>
    <dd class="text-zinc-300">{crop.class_labeler}</dd>
  {/if}
</dl>

<!-- Reprocess: absent unless the backend serves it. -->
<div class="mt-2">
  <ReprocessControl
    target={{ kind: 'crop', cropId: crop.id }}
    onadopt={(crops) => {
      const updated = crops.find((c) => c.id === crop.id);
      if (updated) onreprocessed?.(updated);
    }}
  />
</div>

<CropMetaSlots {crop} />

<CropItemText {crop} />

<CropHistoryBlock
  historyEntries={loader.historyEntries}
  historyError={loader.historyError}
  historyLoading={loader.historyLoading}
/>

<CropSourceContext {crop} {loader} />
