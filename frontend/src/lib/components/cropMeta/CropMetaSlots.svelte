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

  // DQ-m7: this used to be
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
    <div class="mt-4 border-t border-line pt-3">
      <div class="mb-1.5 text-meta text-fg-2 font-semibold">
        {spec.label.title}
      </div>
      <dl class="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-meta">
        <!-- R7 (visual audit 2026-09-24): the served status label first,
             so Details and the review panel's status dropdown use one
             vocabulary; the slot profile's own label only as a fallback. -->
        {#if data?.lifecycle?.status || data?.lifecycle?.state}
          <dt class="text-fg-3">Status</dt>
          <dd class="text-fg-1">
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
          <dt class="text-fg-3">Validation</dt>
          <dd class="flex flex-wrap items-center gap-1.5">
            {#if data.lifecycle.validated}
              <span
                class="rounded border border-success/40 bg-success/15 px-1.5 py-0.5 text-caption text-success-text"
              >
                human validated
              </span>
            {:else if data.lifecycle.autoConfirmed}
              <span
                class="rounded border border-accent/40 bg-accent/15 px-1.5 py-0.5 text-caption text-accent-text"
              >
                auto-confirmed (unreviewed)
              </span>
            {:else if data.lifecycle.validated != null || data.lifecycle.autoConfirmed != null}
              <span class="text-fg-3">not yet reviewed</span>
            {/if}
          </dd>
        {/if}

        {#if data?.subBox?.score != null}
          <dt class="text-fg-3">Score</dt>
          <dd class="tabular-nums">{pct(data.subBox.score)}</dd>
        {/if}

        {#if data?.boxSet && data.boxSet.setComplete === false}
          <dt class="text-fg-3">Box set</dt>
          <dd>
            <span
              class="rounded border border-warn/40 bg-warn/15 px-1.5 py-0.5 text-caption text-warn-text"
              title="The VLM reported visible regions that are missing from this list"
            >
              incomplete
            </span>
          </dd>
        {/if}

        {#if data?.subBoxes && data.subBoxes.length > 0}
          <!-- One block per box, numbered by its current position in the
               served list (the id, not the number, is stable). -->
          <dt class="text-fg-3">Boxes ({data.subBoxes.length})</dt>
          <dd class="flex flex-col gap-1.5">
            {#each data.subBoxes as b, i (b.boxId ?? i)}
              {@const rejectionKind = regionVocabularyStore.rejectionReasonKind(
                b.rejectionReason,
              )}
              <div class="flex flex-col gap-0.5" data-testid="meta-box">
                <div class="flex flex-wrap items-center gap-1.5">
                  <span class="text-caption text-fg-3 tabular-nums">#{i + 1}</span>
                  <span
                    class="rounded border px-1 py-0.5 text-caption {toneChipClass(
                      regionStatusesStore.boxStateTone(b.state),
                    )}"
                  >
                    {regionStatusesStore.boxStateInfo(b.state)?.label ?? b.state}
                  </span>
                  {#if b.score != null}
                    <span class="text-caption text-fg-3 tabular-nums">{pct(b.score)}</span
                    >
                  {/if}
                  {#if b.locked}
                    <span
                      class="rounded border border-line-strong bg-surface-2 px-1 text-caption text-fg-2"
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
                      class="rounded border border-danger/40 bg-danger/15 px-1 text-caption text-danger-text"
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
                    class="text-caption {rejectionKind === 'model_verdict'
                      ? 'text-danger-text'
                      : rejectionKind === 'needs_human'
                        ? 'text-fg-2'
                        : 'text-warn-text'}"
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
                      class="rounded border border-line-strong bg-surface-1 px-1.5 py-0.5 text-fg-1"
                    >
                      {b.text || '∅'}
                    </span>
                    {#if b.textSource}
                      <ProvenanceChip detector={b.textSource} size="sm" />
                    {/if}
                    <span class="text-caption text-fg-3 tabular-nums">
                      {pct(b.textConfidence)}
                    </span>
                    {#if b.textDisagreement}
                      <span
                        class="whitespace-nowrap rounded border border-warn/40 bg-warn/15 px-1 text-caption text-warn-text"
                        title="The VLM and OCR readers disagree on this text"
                      >
                        readers disagree
                      </span>
                    {/if}
                  </div>
                {/if}
                {#if b.textVlm != null || b.textOcr != null}
                  <div class="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-caption">
                    {#if b.textVlm != null}
                      <span>
                        <span class="text-fg-3">vlm:</span>
                        <span class="text-fg-1">{b.textVlm || '∅'}</span>
                      </span>
                    {/if}
                    {#if b.textOcr != null}
                      <span>
                        <span class="text-fg-3">ocr:</span>
                        <span class="text-fg-1">{b.textOcr || '∅'}</span>
                      </span>
                    {/if}
                  </div>
                {/if}
                {#if b.textReader}
                  <div class="text-caption text-fg-3">
                    text reader {b.textReader}{b.textReaderVersion
                      ? ` ${b.textReaderVersion}`
                      : ''}
                  </div>
                {/if}
                <!-- Why the chosen reading won / why the VLM's own reading
                     was rejected as not text (served vocabulary ids). -->
                {#if b.textChoice && b.textChoice !== 'human'}
                  <div class="text-caption text-fg-2">
                    Text choice: {regionVocabularyStore.textChoiceLabel(b.textChoice)}
                  </div>
                {/if}
                {#if b.textVlmInvalid}
                  <div class="text-caption text-danger-text">
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
          <dt class="text-fg-3">Detector</dt>
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
          <dt class="text-fg-3">Cascade</dt>
          <dd class="flex flex-wrap items-center gap-1">
            {#each data.provenance.chain as entry, i (i)}
              <ProvenanceChip raw={entry} size="sm" />
            {/each}
          </dd>
        {/if}

        {#if data?.text?.value != null}
          <dt class="text-fg-3">Text</dt>
          <dd class="flex min-w-0 flex-wrap items-center gap-1.5">
            <span
              class="rounded border border-line-strong bg-surface-1 px-1.5 py-0.5 text-fg-1"
            >
              {data.text.value || '∅'}
            </span>
            {#if data.text.source}
              <ProvenanceChip detector={data.text.source} size="sm" />
            {/if}
            <span class="text-caption text-fg-3 tabular-nums">
              {pct(data.text.confidence)}
            </span>
          </dd>
        {/if}

        {#if data?.text?.engineVersion}
          <dt class="text-fg-3">Text engine</dt>
          <dd class="text-fg-2">{data.text.engineVersion}</dd>
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
          <dt class="text-fg-3">
            {rejectionKind === 'needs_human' ? 'Needs review' : 'Rejection'}
          </dt>
          <dd
            class={rejectionKind === 'model_verdict'
              ? 'text-danger-text'
              : rejectionKind === 'needs_human'
                ? 'text-fg-2'
                : 'text-warn-text'}
          >
            {regionVocabularyStore.rejectionReasonLabel(data.lifecycle.rejectionReason)}
          </dd>
        {/if}
      </dl>
    </div>
  {/if}
{/each}
