<!--
  The /review slot-tab panel: score, status, detector and cascade
  provenance, the reading and rejection-reason editors, and the slot
  actions with the step-back footer.
-->
<script lang="ts">
  import AssistButton from '$lib/components/assistRun/AssistButton.svelte';
  import AssistSuggestions from '$lib/components/assistRun/AssistSuggestions.svelte';
  import ProvenanceChip from '$lib/components/ProvenanceChip.svelte';
  import RejectedBoxChips from '$lib/components/review/RejectedBoxChips.svelte';
  import { slotOf } from '$lib/annotations/cropSlots';
  import type { SlotSpec } from '$lib/annotations/types';
  import { kg } from '$lib/review/reviewKeys';
  import type { ReviewSlotActions } from '$lib/review/reviewSlotActions.svelte';
  import type { ReviewSlotEditController } from '$lib/review/reviewSlotEditController.svelte';
  import {
    statusWantsRejectionReason,
    type SlotPanelLabels,
  } from '$lib/review/slotPanel';
  import type { ReviewItem } from '$lib/types_review';
  import { regionStatusesStore } from '$stores/regionStatuses.svelte';
  import { regionVocabularyStore } from '$stores/regionVocabulary.svelte';

  interface Props {
    current: ReviewItem;
    activeSlot: SlotSpec;
    slotLabels: SlotPanelLabels;
    edit: ReviewSlotEditController;
    slot: ReviewSlotActions;
    skip: () => void;
  }

  let { current, activeSlot, slotLabels, edit, slot, skip }: Props = $props();

  const multiBox = $derived(edit.multiBox);
  const isMultiBoxSlot = $derived(edit.isMultiBoxSlot);
  const editMode = $derived(edit.editMode);
  const selectedSlotBox = $derived(edit.selectedSlotBox);
  const slotStatusOptions = $derived(edit.slotStatusOptions);
  const editedSlotStatus = $derived(edit.editedSlotStatus);
  const editedRejectionReason = $derived(edit.editedRejectionReason);
  const commitSlotStatus = $derived(edit.commitSlotStatus);
  const commitSlotText = $derived(edit.commitSlotText);
  const commitRejectionReason = $derived(edit.commitRejectionReason);
  const toggleEdit = $derived(edit.toggleEdit);
  const slotUndoStack = $derived(slot.slotUndoStack);
  const confirmSlot = $derived(slot.confirmSlot);
  const confirmMultiBoxSlot = $derived(slot.confirmMultiBoxSlot);
  const rejectSlot = $derived(slot.rejectSlot);
  const markFalsePositive = $derived(slot.markFalsePositive);
  const slotBack = $derived(slot.slotBack);
  const slotData = $derived(slotOf(current, activeSlot));
</script>

<!-- Slot inline review. The canvas above is live — drag/resize
   the proposal in place and hit Enter to confirm. The Reject
   button (or D) marks the slot's rejectState. The whole flow
   is two keystrokes per crop on average: minor twitch with
   arrows / handles, then Enter. -->
<div class="mt-3 grid grid-cols-2 gap-y-1 text-xs">
  <span class="text-zinc-500">{slotLabels.scoreLabel}</span>
  <span class="font-mono text-zinc-200">
    {#if isMultiBoxSlot && selectedSlotBox?.score != null}
      {(selectedSlotBox.score * 100).toFixed(1)}%
    {:else if !isMultiBoxSlot && slotData?.subBox?.score != null}
      {(slotData.subBox.score * 100).toFixed(1)}%
    {:else}
      —
    {/if}
  </span>
  <span class="text-zinc-500">{slotLabels.statusLabel}</span>
  <span class="flex items-center gap-1.5">
    <select
      bind:value={edit.editedSlotStatus}
      onchange={() => void commitSlotStatus()}
      class="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-xs text-zinc-100 focus:border-blue-500 focus:outline-none"
    >
      <option value="">—</option>
      {#each slotStatusOptions as opt (opt.value)}
        <option value={opt.value}>{opt.label}</option>
      {/each}
    </select>
    <!-- dq-region (2026-09-24): validated (human-only) vs
         auto_confirmed (machine accepted, unreviewed) —
         region_verified no longer distinguishes the two. -->
    {#if slotData?.lifecycle?.validated}
      <span
        class="rounded border border-emerald-500/40 bg-emerald-500/15 px-1 text-[10px] text-emerald-200"
        title="A human confirmed/drew/rejected this region"
      >
        human validated
      </span>
    {:else if slotData?.lifecycle?.autoConfirmed}
      <span
        class="rounded border border-blue-500/40 bg-blue-500/15 px-1 text-[10px] text-blue-200"
        title="Auto-confirmed by the worker's policy — accepted but not yet reviewed by a human"
      >
        auto-confirmed
      </span>
    {/if}
    <!-- The verifier's own box-correctness verdict for the
         selected box: false is the "model said wrong box"
         signal. -->
    {#if selectedSlotBox?.bboxCorrect === false}
      <span
        class="rounded border border-red-500/40 bg-red-500/15 px-1 text-[10px] text-red-200"
        title="The verifier judged this box incorrect"
      >
        model: box wrong
      </span>
    {/if}
    {#if selectedSlotBox?.locked}
      <span
        class="rounded border border-zinc-600 bg-zinc-800 px-1 text-[10px] text-zinc-300"
        title="A human created or edited this box; automated stages leave it alone"
      >
        locked
      </span>
    {/if}
    {#if slotData?.boxSet?.setComplete === false}
      <span
        class="rounded border border-amber-500/40 bg-amber-500/15 px-1 text-[10px] text-amber-200"
        title="The VLM reported visible regions that are missing from this list"
      >
        set incomplete
      </span>
    {/if}
  </span>
  <span class="text-zinc-500">Detector</span>
  <span class="flex flex-wrap items-center gap-1.5">
    {#if selectedSlotBox?.detector || slotData?.provenance?.detector}
      <ProvenanceChip
        detector={selectedSlotBox?.detector ?? slotData?.provenance?.detector ?? null}
        version={selectedSlotBox?.detectorVersion ??
          slotData?.provenance?.detectorVersion ??
          null}
      />
    {:else}
      <span class="text-zinc-500">—</span>
    {/if}
    {#if slotData?.provenance?.verifier}
      <ProvenanceChip
        detector={slotData.provenance.verifier}
        tag="verify"
        version={slotData.provenance.verifierVersion}
        size="sm"
      />
    {/if}
    <!-- The mistakenness chip lives in the Scores row above, not
         next to the detector provenance chips. -->
    {#if isMultiBoxSlot && activeSlot}
      <!-- No separate "candidate" concept — a
           verifier-rejected box is just a SlotBox with
           state: 'rejected' and its own rejectionReason
           (owner decision 2026-09-26, no backward
           compatibility). Show a needs-review/rejected chip
           per rejected box, styled by the served kind, same
           as before — worded per-box, not per-item. -->
      <RejectedBoxChips boxes={multiBox.boxes} subBoxes={slotData?.subBoxes} />
      {#if multiBox.boxes.length === 0 && !editMode}
        <span
          class="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-[10px] text-zinc-400"
        >
          no boxes · press {kg('review.region.edit_box')} to draw
        </span>
      {/if}
    {/if}
  </span>
  {#if isMultiBoxSlot && selectedSlotBox?.boxId}
    <span class="text-zinc-500">Assist</span>
    <span class="flex flex-wrap items-center gap-2" data-testid="review-box-assist">
      <AssistSuggestions assist={current.assist} boxId={selectedSlotBox.boxId} />
      <AssistButton
        label="Assist on this box…"
        opener={{
          title: 'This box',
          selected: {
            kind: 'boxes',
            boxes: [{ crop_id: current.id, box_id: selectedSlotBox.boxId }],
          },
        }}
        onclosed={() => void slot.reloadCurrent()}
      />
    </span>
  {/if}
  {#if slotData?.provenance?.chain && slotData.provenance.chain.length > 0}
    <span class="text-zinc-500">Cascade</span>
    <span class="flex flex-wrap items-center gap-1">
      {#each slotData.provenance.chain as entry, i (i)}
        <ProvenanceChip raw={entry} size="sm" />
      {/each}
    </span>
  {/if}
  <!-- Text-free regions: a slot with no
       text capability renders no text row/edit at all — not
       even a disabled input — since the backend 422s
       `region_text` on such a profile. -->
  {#if activeSlot.capabilities.text}
    <span class="text-zinc-500">{slotLabels.textLabel}</span>
    <span class="flex items-center gap-1.5">
      <input
        type="text"
        bind:value={edit.editedSlotText}
        onblur={() => void commitSlotText()}
        onkeydown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            (e.currentTarget as HTMLInputElement).blur();
          }
        }}
        placeholder={slotLabels.textPlaceholder}
        disabled={isMultiBoxSlot && selectedSlotBox == null}
        spellcheck="false"
        autocapitalize={activeSlot.capabilities.text?.transform === 'uppercase'
          ? 'characters'
          : 'off'}
        class="w-28 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-xs text-zinc-100 placeholder:italic placeholder:text-zinc-600 focus:border-blue-500 focus:outline-none {activeSlot
          .capabilities.text?.monospace
          ? 'font-mono'
          : ''}"
      />
      {#if isMultiBoxSlot ? selectedSlotBox?.textSource : slotData?.text?.source}
        <ProvenanceChip
          detector={(isMultiBoxSlot
            ? selectedSlotBox?.textSource
            : slotData?.text?.source) ?? null}
          size="sm"
        />
      {/if}
      {#if (isMultiBoxSlot ? selectedSlotBox?.textConfidence : slotData?.text?.confidence) != null}
        <span class="text-[10px] text-zinc-500">
          {(
            ((isMultiBoxSlot
              ? selectedSlotBox?.textConfidence
              : slotData?.text?.confidence) ?? 0) * 100
          ).toFixed(0)}%
        </span>
      {/if}
      {#if selectedSlotBox?.textDisagreement}
        <span
          class="rounded border border-orange-500/40 bg-orange-500/15 px-1 text-[10px] text-orange-200"
          title="vlm: {selectedSlotBox.textVlm ?? '∅'} · ocr: {selectedSlotBox.textOcr ??
            '∅'}"
        >
          readers disagree
        </span>
      {/if}
      <!-- Why the chosen reading won / why the VLM's own
           reading was rejected as not text (served
           vocabulary ids). -->
      {#if selectedSlotBox?.textChoice && selectedSlotBox.textChoice !== 'human'}
        <span
          class="rounded border border-zinc-700 bg-zinc-900 px-1 text-[10px] text-zinc-400"
          title="How this reading was chosen"
        >
          {regionVocabularyStore.textChoiceLabel(selectedSlotBox.textChoice)}
        </span>
      {/if}
      {#if selectedSlotBox?.textVlmInvalid}
        <span
          class="rounded border border-red-500/40 bg-red-500/15 px-1 text-[10px] text-red-200"
          title="Why the VLM's own reading wasn't used as text"
        >
          vlm invalid: {regionVocabularyStore.invalidReasonLabel(
            selectedSlotBox.textVlmInvalid,
          )}
        </span>
      {/if}
    </span>
  {/if}
  {#if statusWantsRejectionReason(activeSlot, editedSlotStatus, regionStatusesStore.list)}
    {@const servedReasonKind = regionVocabularyStore.rejectionReasonKind(
      slotData?.lifecycle?.rejectionReason,
    )}
    <span class="text-zinc-500">Rejection reason</span>
    <span>
      {#if servedReasonKind && editedRejectionReason === (slotData?.lifecycle?.rejectionReason ?? '')}
        <!-- R6 (visual audit 2026-09-24): a machine reason from the
             served vocabulary (e.g. verifier_no_verdict) used to
             sit raw in an editable box; show its served label
             instead. Changing the status still re-asks. -->
        <span class="text-zinc-300" data-testid="served-rejection-reason"
          >{regionVocabularyStore.rejectionReasonLabel(
            slotData?.lifecycle?.rejectionReason,
          )}</span
        >
      {:else}
        <input
          type="text"
          bind:value={edit.editedRejectionReason}
          onblur={() => void commitRejectionReason()}
          onkeydown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              (e.currentTarget as HTMLInputElement).blur();
            }
          }}
          placeholder="e.g. blurred, occluded, glare"
          class="w-44 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-xs text-zinc-100 focus:border-blue-500 focus:outline-none"
        />
      {/if}
    </span>
  {/if}
</div>
<div
  class="sticky bottom-0 z-10 mt-3 flex flex-wrap gap-2 bg-[rgb(var(--bg-elevated))] py-1"
  data-testid="review-actions"
>
  {#if editMode}
    <button
      class="btn btn-primary"
      type="button"
      onclick={confirmMultiBoxSlot}
      disabled={multiBox.busy}
    >
      Save boxes
    </button>
    <button class="btn" type="button" onclick={toggleEdit} disabled={multiBox.busy}>
      Cancel
    </button>
  {:else}
    <button
      class="btn btn-primary"
      type="button"
      onclick={isMultiBoxSlot ? confirmMultiBoxSlot : confirmSlot}
    >
      {slotLabels.confirmLabel}
    </button>
    <button class="btn btn-danger" type="button" onclick={rejectSlot}>
      {slotLabels.rejectLabel}
    </button>
    {#if activeSlot.capabilities.lifecycle?.falsePositiveState}
      <button
        class="btn"
        type="button"
        onclick={markFalsePositive}
        title="Detector drew a box but it's not the {activeSlot.label
          .singular} — keep the box as a training hard negative ({kg(
          'review.region.false_positive',
        )})"
      >
        False positive
      </button>
    {/if}
    <button class="btn" type="button" onclick={skip}>Skip</button>
    {#if isMultiBoxSlot}
      <button
        class="btn"
        type="button"
        onclick={toggleEdit}
        aria-pressed={editMode}
        title="Toggle bbox edit mode ({kg('review.region.edit_box')})"
      >
        Edit boxes
      </button>
    {/if}
    <button
      class="btn"
      type="button"
      onclick={slotBack}
      disabled={slotUndoStack.length === 0}
      title="Re-open the most-recently confirmed {activeSlot.label.singular} ({kg(
        'review.region.back',
      )}) — only re-queues it locally, use {kg('review.undo')} to undo the server write"
    >
      ← Step back
    </button>
  {/if}
</div>
{#if slotUndoStack.length > 0}
  <!-- DQ-m6 (docs/design/data-quality-pass-2026-09-24.md):
       this stack holds confirm, reject AND false-positive
       entries — the
       footer said "confirmed" unconditionally, so a reject
       read as "1 confirmed in this session". "Actioned" is
       accurate for all three. -->
  <p class="mt-1 text-[10px] text-zinc-500">
    {slotUndoStack.length} actioned in this session — press {kg('review.region.back')} to step
    back.
  </p>
{/if}
