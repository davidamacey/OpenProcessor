<script lang="ts">
  import type { Crop, CropHistoryEntry, CropImageMeta } from '$lib/types';
  import { getCropHistory, getCropContext, getThumbUrl } from '$lib/api';
  import ProvenanceChip from './ProvenanceChip.svelte';
  import { slotRegistry } from '$lib/annotations/registeredSlots';
  import { slotOf } from '$lib/annotations/cropSlots';
  import { slotIsPresent } from '$lib/annotations/types';
  import { classSourcesStore } from '$stores/classSources.svelte';

  interface Props {
    crop: Crop;
  }

  let { crop }: Props = $props();

  const vlmConf = $derived<string | null>(crop.vlm_confidence ?? null);
  const classSource = $derived<string | null>(crop.class_source ?? null);
  // DQ-M8: served role, not a hardcoded string match against class_source
  // — the catalog (GET {API_PREFIX}/class_sources, classSourcesStore) is
  // the deployment's actual vocabulary (sourceBadge.ts uses the same
  // role.startsWith('vlm') check for the label-source badge).
  const isVlmSourced = $derived(
    (classSourcesStore.roleFor(classSource) ?? '').startsWith('vlm'),
  );

  // DQ-m7 (docs/design/data-quality-pass-2026-09-24.md): this used to be
  // `slotRegistry.forClass(crop.class_id, ...)` — slots bound to the
  // crop's OWN class. That's the wrong question for a sub-box slot like
  // license_plate: a plate box lives on a VEHICLE crop (class "sedan",
  // "suv", ...), never on a crop literally classified "license_plate", so
  // forClass() here was structurally guaranteed to return nothing for
  // every real plate-bearing crop — the modal showed item text and class
  // history but never the plate's status, chosen text, VLM/OCR
  // candidates, disagreement flag or detector chain, on every crop
  // (00000000 included) regardless of whether it actually had plate
  // data. The correct question is "does this crop carry evidence for
  // this slot" (slotIsPresent(slotOf(crop, spec))) — the same check
  // every other slot-generic surface in this app uses (review/+page.svelte,
  // SlotCard.svelte) — evaluated over every REGISTERED slot, not just the
  // one (if any) bound to the crop's own class.
  const presentSlots = $derived(
    slotRegistry.all.filter((spec) => slotIsPresent(slotOf(crop, spec))),
  );

  function pct(value: number | null | undefined): string {
    return value == null ? '—' : `${(value * 100).toFixed(1)}%`;
  }

  // Served verbatim, never inferred client-side: 'vlm' | 'model' | null.
  const classConfidenceSourceLabel = $derived<string>(
    crop.class_confidence_source === 'vlm'
      ? 'VLM'
      : crop.class_confidence_source === 'model'
        ? 'Model'
        : '',
  );

  // -- History (G7) — lazy, fetched once per crop.id, never blocks the
  // rest of the panel from rendering. ------------------------------------
  let historyEntries = $state<CropHistoryEntry[] | null>(null);
  let historyError = $state<string | null>(null);
  let historyLoading = $state(false);

  // -- Source image + siblings (G7/G8) — same lazy-on-open pattern. ------
  let imageMeta = $state<CropImageMeta | null>(null);
  let siblings = $state<Crop[] | null>(null);
  let imageError = $state<string | null>(null);
  let imageLoading = $state(false);

  let showTextBoxes = $state(false);

  $effect(() => {
    const id = crop.id;
    historyEntries = null;
    historyError = null;
    historyLoading = true;
    const controller = new AbortController();
    getCropHistory(id, controller.signal)
      .then((res) => {
        historyEntries = res.entries;
      })
      .catch((e: unknown) => {
        if ((e as Error)?.name === 'AbortError') return;
        historyError = (e as Error).message;
      })
      .finally(() => {
        historyLoading = false;
      });
    return () => controller.abort();
  });

  $effect(() => {
    const id = crop.id;
    imageMeta = null;
    siblings = null;
    imageError = null;
    imageLoading = true;
    const controller = new AbortController();
    getCropContext(id, controller.signal)
      .then((res) => {
        imageMeta = res.image;
        siblings = res.items.filter((it) => it.id !== id);
      })
      .catch((e: unknown) => {
        if ((e as Error)?.name === 'AbortError') return;
        imageError = (e as Error).message;
      })
      .finally(() => {
        imageLoading = false;
      });
    return () => controller.abort();
  });
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
      <span class="ml-1 font-mono text-amber-300/60">{crop.excluded_at}</span>
    {/if}
  </div>
{/if}

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
    {#if crop.class_validated}
      <span
        class="ml-1 rounded border border-green-500/40 bg-green-500/15 px-1 text-[10px] text-green-200"
      >
        validated
      </span>
    {/if}
  </dd>

  <!-- DQ-M8 (docs/design/data-quality-pass-2026-09-24.md): `label_confidence`
       (wire `confidence`) is the vehicle-detector/v6 score on EVERY row,
       including ones the VLM labeled — never the VLM's own confidence.
       Calling it plain "Confidence" next to a VLM-sourced label reads as
       the VLM's certainty (the design doc's repro: "Confidence 94.6%"
       under "Current label dumptruck (vlm)"). Label it for what it is
       whenever the class came from the VLM, and show the VLM's own
       categorical confidence (`vlm_confidence`, served separately) as
       its own row instead of folding it in as a same-row detail. -->
  <dt class="text-zinc-500">{isVlmSourced ? 'Detector score' : 'Confidence'}</dt>
  <dd class="font-mono">{pct(crop.label_confidence)}</dd>

  {#if vlmConf}
    <dt class="text-zinc-500">VLM confidence</dt>
    <dd class="font-mono text-zinc-200">{vlmConf}</dd>
  {/if}

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

  {#if crop.vlm_class_empty_reason}
    <dt class="text-zinc-500">VLM empty reason</dt>
    <dd class="text-orange-300">{crop.vlm_class_empty_reason}</dd>
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
    <dd class="font-mono break-all text-zinc-400">{crop.class_labeled_at}</dd>
  {/if}

  {#if crop.class_labeler}
    <dt class="text-zinc-500">Labeler</dt>
    <dd class="text-zinc-300">{crop.class_labeler}</dd>
  {/if}
</dl>

{#each presentSlots as spec (spec.key)}
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
            {#if data.text.disagreement}
              <span
                class="rounded border border-orange-500/40 bg-orange-500/15 px-1 text-[10px] text-orange-200"
                title="The VLM and OCR readers disagree on this text"
              >
                readers disagree
              </span>
            {/if}
          </dd>
        {/if}

        {#if data?.text?.vlmValue != null || data?.text?.ocrValue != null}
          <dt class="text-zinc-500">Candidates</dt>
          <dd class="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px]">
            {#if data.text.vlmValue != null}
              <span>
                <span class="text-zinc-500">vlm:</span>
                <span class="font-mono text-zinc-200">{data.text.vlmValue || '∅'}</span>
              </span>
            {/if}
            {#if data.text.ocrValue != null}
              <span>
                <span class="text-zinc-500">ocr:</span>
                <span class="font-mono text-zinc-200">{data.text.ocrValue || '∅'}</span>
              </span>
            {/if}
          </dd>
        {/if}

        {#if data?.text?.engineVersion}
          <dt class="text-zinc-500">Text engine</dt>
          <dd class="font-mono text-zinc-400">{data.text.engineVersion}</dd>
        {/if}

        {#if data?.lifecycle?.rejectionReason}
          <dt class="text-zinc-500">Rejection</dt>
          <dd class="text-zinc-300">{data.lifecycle.rejectionReason}</dd>
        {/if}
      </dl>
    </div>
  {/if}
{/each}

{#if crop.item_text_lines && crop.item_text_lines.length > 0}
  <div class="mt-4 border-t border-zinc-800 pt-3">
    <div
      class="mb-1.5 flex items-center justify-between text-[10px] uppercase tracking-wider text-zinc-500"
    >
      <span>Item text</span>
      <button
        type="button"
        class="lowercase tracking-normal text-zinc-400 hover:text-zinc-200"
        onclick={() => (showTextBoxes = !showTextBoxes)}
      >
        {showTextBoxes ? 'hide boxes' : 'show boxes'}
      </button>
    </div>
    {#if showTextBoxes}
      <div class="relative mb-2 aspect-square w-full overflow-hidden rounded bg-zinc-900">
        <img
          src={getThumbUrl(crop.id, 320)}
          alt="item text overlay"
          class="h-full w-full object-contain"
        />
        {#each crop.item_text_lines as line, i (i)}
          {#if line.box_norm && line.box_norm.length === 4}
            {@const [x1, y1, x2, y2] = line.box_norm}
            <div
              class="pointer-events-none absolute border border-cyan-400/80"
              style="left:{x1 * 100}%; top:{y1 * 100}%; width:{(x2 - x1) *
                100}%; height:{(y2 - y1) * 100}%;"
            ></div>
          {/if}
        {/each}
      </div>
    {/if}
    <ul class="space-y-0.5 text-xs">
      {#each crop.item_text_lines as line, i (i)}
        <li class="flex items-center gap-1.5">
          <span
            class="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 font-mono text-zinc-100"
          >
            {line.text || '∅'}
          </span>
          {#if line.confidence != null}
            <span class="font-mono text-[10px] text-zinc-500">{pct(line.confidence)}</span
            >
          {/if}
        </li>
      {/each}
    </ul>
  </div>
{/if}

<div class="mt-4 border-t border-zinc-800 pt-3">
  <div class="mb-1.5 text-[10px] uppercase tracking-wider text-zinc-500">History</div>
  {#if historyLoading}
    <p class="text-[11px] text-zinc-500">Loading…</p>
  {:else if historyError}
    <p class="text-[11px] text-red-300">History unavailable: {historyError}</p>
  {:else if !historyEntries || historyEntries.length === 0}
    <p class="text-[11px] text-zinc-500">No prior writes recorded.</p>
  {:else}
    <ul class="space-y-1 text-[11px]">
      {#each historyEntries as entry, i (i)}
        <li class="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-zinc-400">
          <span class="font-mono text-zinc-300">{entry.writer ?? 'unknown'}</span>
          {#if entry.class_name}
            <span>→ {entry.class_name}</span>
          {/if}
          {#if entry.class_source}
            <span class="text-zinc-600">({entry.class_source})</span>
          {/if}
          {#if entry.at}
            <span class="font-mono text-zinc-600">{entry.at}</span>
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</div>

<div class="mt-4 border-t border-zinc-800 pt-3">
  <div class="mb-1.5 text-[10px] uppercase tracking-wider text-zinc-500">
    Source image
  </div>
  {#if imageLoading}
    <p class="text-[11px] text-zinc-500">Loading…</p>
  {:else if imageError}
    <p class="text-[11px] text-red-300">Source image unavailable: {imageError}</p>
  {:else if imageMeta}
    <dl class="grid grid-cols-2 gap-y-1 text-[11px] text-zinc-400">
      {#if imageMeta.width != null && imageMeta.height != null}
        <dt class="text-zinc-500">Size</dt>
        <dd class="font-mono">{imageMeta.width}×{imageMeta.height}</dd>
      {/if}
      {#if imageMeta.source}
        <dt class="text-zinc-500">Source</dt>
        <dd>{imageMeta.source}</dd>
      {/if}
      {#if imageMeta.indexed_at}
        <dt class="text-zinc-500">Indexed</dt>
        <dd class="font-mono">{imageMeta.indexed_at}</dd>
      {/if}
    </dl>
    {#if siblings && siblings.length > 0}
      <div class="mt-2 text-[10px] uppercase tracking-wider text-zinc-500">
        Siblings ({siblings.length})
      </div>
      <div class="mt-1 flex flex-wrap gap-1.5">
        {#each siblings as sib (sib.id)}
          <img
            src={getThumbUrl(sib.id, 64)}
            alt="sibling crop"
            title={sib.class_name ?? sib.id}
            class="h-12 w-12 rounded border border-zinc-800 bg-zinc-900 object-contain"
          />
        {/each}
      </div>
    {/if}
  {/if}
</div>

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
