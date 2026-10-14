<script lang="ts">
  import type { Crop } from '$lib/types';
  import ProvenanceChip from '$components/ProvenanceChip.svelte';
  import { slotRegistry } from '$lib/annotations/registeredSlots';
  import { slotOf } from '$lib/annotations/cropSlots';
  import { slotIsPresent } from '$lib/annotations/types';
  import { regionVocabularyStore } from '$stores/regionVocabulary.svelte';
  import { regionStatusesStore, toneChipClass } from '$stores/regionStatuses.svelte';
  import { pct } from '$lib/cropMeta/pct';

  let { crop }: { crop: Crop } = $props();

  // DQ-m7 (docs/design/data-quality-pass-2026-09-24.md): this used to be
  // `slotRegistry.forClass(crop.class_id, ...)` — slots bound to the
  // crop's OWN class. That's the wrong question for a sub-box slot: a
  // region (e.g. a tag) lives on an item of another class (e.g.
  // "widget_a"), never on a crop classified as the region's own class, so
  // forClass() here was structurally guaranteed to return nothing for
  // every real region-bearing crop — the modal showed item text and class
  // history but never the region's status, chosen text, VLM/OCR
  // candidates, disagreement flag or detector chain, regardless of
  // whether the crop actually had region data. The correct question is "does this crop carry evidence for
  // this slot" (slotIsPresent(slotOf(crop, spec))) — the same check
  // every other slot-generic surface in this app uses (review/+page.svelte,
  // SlotCard.svelte) — evaluated over every REGISTERED slot, not just the
  // one (if any) bound to the crop's own class.
  const presentSlots = $derived(
    slotRegistry.all.filter((spec) => slotIsPresent(slotOf(crop, spec))),
  );
</script>

{#each presentSlots as spec (spec.key)}
  {@const data = slotOf(crop, spec)}
  {#if slotIsPresent(data)}
    <div class="mt-4 border-t border-zinc-800 pt-3">
      <div class="mb-1.5 text-[10px] uppercase tracking-wider text-zinc-500">
        {spec.label.title}
      </div>
      <dl class="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-xs">
        <!-- R7 (visual audit 2026-09-24): the served status label first,
             so Details and the review panel's status dropdown use one
             vocabulary; the slot profile's own label only as a fallback. -->
        {#if data?.lifecycle?.status || data?.lifecycle?.state}
          <dt class="text-zinc-500">Status</dt>
          <dd class="text-zinc-200">
            {regionStatusesStore.labelFor(data.lifecycle.status) ??
              data.lifecycle.state?.label ??
              data.lifecycle.status}
          </dd>
        {/if}

        <!-- dq-region (2026-09-24): human-validated (region_validated,
             now human-only) vs auto-confirmed (region_auto_confirmed,
             machine-accepted but not yet reviewed). Neither reuses
             region_verified, whose meaning ("a verification pass ran")
             is now a third, distinct thing. -->
        {#if data?.lifecycle?.validated != null || data?.lifecycle?.autoConfirmed != null}
          <dt class="text-zinc-500">Validation</dt>
          <dd class="flex flex-wrap items-center gap-1.5">
            {#if data.lifecycle.validated}
              <span
                class="rounded border border-emerald-500/40 bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-200"
              >
                human validated
              </span>
            {:else if data.lifecycle.autoConfirmed}
              <span
                class="rounded border border-blue-500/40 bg-blue-500/15 px-1.5 py-0.5 text-[10px] text-blue-200"
              >
                auto-confirmed (unreviewed)
              </span>
            {:else if data.lifecycle.validated != null || data.lifecycle.autoConfirmed != null}
              <span class="text-zinc-500">not yet reviewed</span>
            {/if}
          </dd>
        {/if}

        {#if data?.subBox?.score != null}
          <dt class="text-zinc-500">Score</dt>
          <dd class="font-mono">{pct(data.subBox.score)}</dd>
        {/if}

        {#if data?.boxSet && data.boxSet.setComplete === false}
          <dt class="text-zinc-500">Box set</dt>
          <dd>
            <span
              class="rounded border border-amber-500/40 bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-200"
              title="The VLM reported visible regions that are missing from this list"
            >
              incomplete
            </span>
          </dd>
        {/if}

        {#if data?.subBoxes && data.subBoxes.length > 0}
          <!-- One block per box, numbered by its current position in the
               served list (the id, not the number, is stable). -->
          <dt class="text-zinc-500">Boxes ({data.subBoxes.length})</dt>
          <dd class="flex flex-col gap-1.5">
            {#each data.subBoxes as b, i (b.boxId ?? i)}
              {@const rejectionKind = regionVocabularyStore.rejectionReasonKind(
                b.rejectionReason,
              )}
              <div class="flex flex-col gap-0.5" data-testid="meta-box">
                <div class="flex flex-wrap items-center gap-1.5">
                  <span class="font-mono text-[10px] text-zinc-500">#{i + 1}</span>
                  <span
                    class="rounded border px-1 py-0.5 text-[10px] {toneChipClass(
                      regionStatusesStore.boxStateTone(b.state),
                    )}"
                  >
                    {regionStatusesStore.boxStateInfo(b.state)?.label ?? b.state}
                  </span>
                  {#if b.score != null}
                    <span class="font-mono text-[10px] text-zinc-500">{pct(b.score)}</span
                    >
                  {/if}
                  {#if b.locked}
                    <span
                      class="rounded border border-zinc-600 bg-zinc-800 px-1 text-[10px] text-zinc-300"
                      title="A human created or edited this box; automated stages leave it alone"
                    >
                      locked
                    </span>
                  {/if}
                  <!-- The verifier's own box-correctness verdict: false is
                       the actual "model said wrong box" signal, distinct
                       from a rejection reason's kind. -->
                  {#if b.bboxCorrect === false}
                    <span
                      class="rounded border border-red-500/40 bg-red-500/15 px-1 text-[10px] text-red-200"
                    >
                      model: box wrong
                    </span>
                  {/if}
                  {#if b.detector}
                    <ProvenanceChip
                      detector={b.detector}
                      version={b.detectorVersion}
                      size="sm"
                    />
                  {/if}
                </div>
                {#if b.rejectionReason}
                  <div
                    class="text-[11px] {rejectionKind === 'model_verdict'
                      ? 'text-red-300'
                      : rejectionKind === 'needs_human'
                        ? 'text-zinc-300'
                        : 'text-amber-300'}"
                  >
                    {rejectionKind === 'needs_human'
                      ? 'Needs review: '
                      : 'Rejection: '}{regionVocabularyStore.rejectionReasonLabel(
                      b.rejectionReason,
                    )}
                  </div>
                {/if}
                {#if b.text != null}
                  <div class="flex min-w-0 flex-wrap items-center gap-1.5">
                    <span
                      class="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 font-mono text-zinc-100"
                    >
                      {b.text || '∅'}
                    </span>
                    {#if b.textSource}
                      <ProvenanceChip detector={b.textSource} size="sm" />
                    {/if}
                    <span class="font-mono text-[10px] text-zinc-500">
                      {pct(b.textConfidence)}
                    </span>
                    {#if b.textDisagreement}
                      <span
                        class="whitespace-nowrap rounded border border-orange-500/40 bg-orange-500/15 px-1 text-[10px] text-orange-200"
                        title="The VLM and OCR readers disagree on this text"
                      >
                        readers disagree
                      </span>
                    {/if}
                  </div>
                {/if}
                {#if b.textVlm != null || b.textOcr != null}
                  <div class="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px]">
                    {#if b.textVlm != null}
                      <span>
                        <span class="text-zinc-500">vlm:</span>
                        <span class="font-mono text-zinc-200">{b.textVlm || '∅'}</span>
                      </span>
                    {/if}
                    {#if b.textOcr != null}
                      <span>
                        <span class="text-zinc-500">ocr:</span>
                        <span class="font-mono text-zinc-200">{b.textOcr || '∅'}</span>
                      </span>
                    {/if}
                  </div>
                {/if}
                {#if b.textEngineVersion}
                  <div class="font-mono text-[10px] text-zinc-500">
                    text engine {b.textEngineVersion}
                  </div>
                {/if}
                <!-- Why the chosen reading won / why the VLM's own reading
                     was rejected as not text (served vocabulary ids). -->
                {#if b.textChoice && b.textChoice !== 'human'}
                  <div class="text-[11px] text-zinc-300">
                    Text choice: {regionVocabularyStore.textChoiceLabel(b.textChoice)}
                  </div>
                {/if}
                {#if b.textVlmInvalid}
                  <div class="text-[11px] text-red-300">
                    VLM text rejected: {regionVocabularyStore.invalidReasonLabel(
                      b.textVlmInvalid,
                    )}
                  </div>
                {/if}
              </div>
            {/each}
          </dd>
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
            {#each data.provenance.chain as entry, i (i)}
              <ProvenanceChip raw={entry} size="sm" />
            {/each}
          </dd>
        {/if}

        {#if data?.text?.value != null}
          <dt class="text-zinc-500">Text</dt>
          <dd class="flex min-w-0 flex-wrap items-center gap-1.5">
            <span
              class="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 font-mono text-zinc-100"
            >
              {data.text.value || '∅'}
            </span>
            {#if data.text.source}
              <ProvenanceChip detector={data.text.source} size="sm" />
            {/if}
            <span class="font-mono text-[10px] text-zinc-500">
              {pct(data.text.confidence)}
            </span>
          </dd>
        {/if}

        {#if data?.text?.engineVersion}
          <dt class="text-zinc-500">Text engine</dt>
          <dd class="font-mono text-zinc-400">{data.text.engineVersion}</dd>
        {/if}

        {#if data?.lifecycle?.rejectionReason}
          {@const rejectionKind = regionVocabularyStore.rejectionReasonKind(
            data.lifecycle.rejectionReason,
          )}
          <!-- 3f1a11e adoption: style/label by the served kind —
               model_verdict (the verifier judged the box wrong) reads as
               a rejection, automatic (a geometry gate) is neutral/amber,
               needs_human (no verdict given) must never be worded as a
               rejection. -->
          <dt class="text-zinc-500">
            {rejectionKind === 'needs_human' ? 'Needs review' : 'Rejection'}
          </dt>
          <dd
            class={rejectionKind === 'model_verdict'
              ? 'text-red-300'
              : rejectionKind === 'needs_human'
                ? 'text-zinc-300'
                : 'text-amber-300'}
          >
            {regionVocabularyStore.rejectionReasonLabel(data.lifecycle.rejectionReason)}
          </dd>
        {/if}
      </dl>
    </div>
  {/if}
{/each}
