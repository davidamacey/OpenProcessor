<!--
  The /review item metadata rows: reason or rejection, current label,
  proposal, model prediction, new-class flag, confidence, proposal hint,
  rank and clarity, and scores.
-->
<script lang="ts">
  import ConfirmationChip from '$lib/components/labelConfirmation/ConfirmationChip.svelte';
  import DetectorClass from '$lib/components/labelConfirmation/DetectorClass.svelte';
  import ScoreChip from '$lib/components/ScoreChip.svelte';
  import { slotOf } from '$lib/annotations/cropSlots';
  import type { SlotSpec } from '$lib/annotations/types';
  import { NO_OPINION_TEXT } from '$lib/review/probeOpinion';
  import { NO_CLASS_YET, vlmEmptyReasonText } from '$lib/review/reviewCopy';
  import type { ReviewItemActions } from '$lib/review/reviewItemActions.svelte';
  import type { ReviewItem } from '$lib/types';
  import { classSourcesStore } from '$stores/classSources.svelte';
  import { regionVocabularyStore } from '$stores/regionVocabulary.svelte';

  interface Props {
    current: ReviewItem;
    activeSlot: SlotSpec | null;
    isMultiBoxSlot: boolean;
    opinion: ReviewItemActions['opinion'];
    searchScores: Map<string, number>;
    acceptModelClass: () => Promise<void>;
  }

  let {
    current,
    activeSlot,
    isMultiBoxSlot,
    opinion,
    searchScores,
    acceptModelClass,
  }: Props = $props();

  // The authoritative, kind-styled
  // explanation for the "Reason" row — the generic per-item `reason`
  // string (see below) always says "verifier rejected this candidate
  // (…)", wrong wording for a needs_human item. The MACHINE
  // reason lives on each rejected box (`SlotBox.rejectionReason`), not
  // the item (`region_rejection_reason` is the reviewer's free-text
  // note only, per spec) — for a multi-box slot this reads the
  // first rejected box's reason; a tier-2 single-box slot still reads
  // the item-level field. Only falls back to `current.reason` when
  // neither is present (core tabs, e.g. the mismatches preset).
  const currentSlotRejectionReason = $derived<string | null>(
    activeSlot && current
      ? isMultiBoxSlot
        ? (slotOf(current, activeSlot)?.subBoxes?.find((b) => b.state === 'rejected')
            ?.rejectionReason ?? null)
        : (slotOf(current, activeSlot)?.lifecycle?.rejectionReason ?? null)
      : null,
  );
  // DQ-M8: served role (classSourcesStore, GET {API_PREFIX}/class_sources), not
  // a hardcoded string match — mirrors sourceBadge.ts's
  // role.startsWith('vlm') check for the label-source badge.
  const isCurrentLabelVlmSourced = $derived(
    (classSourcesStore.roleFor(current?.label_source) ?? '').startsWith('vlm'),
  );
</script>

<dl class="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
  {#if currentSlotRejectionReason}
    {@const reasonKind = regionVocabularyStore.rejectionReasonKind(
      currentSlotRejectionReason,
    )}
    <dt class="text-zinc-500">
      {reasonKind === 'needs_human' ? 'Needs review' : 'Rejection'}
    </dt>
    <dd
      class={reasonKind === 'model_verdict'
        ? 'text-red-300'
        : reasonKind === 'needs_human'
          ? 'text-zinc-200'
          : 'text-amber-300'}
    >
      {regionVocabularyStore.rejectionReasonLabel(currentSlotRejectionReason)}
    </dd>
  {:else}
    <dt class="text-zinc-500">Reason</dt>
    <dd class="text-zinc-200">{current.reason ?? '—'}</dd>
  {/if}

  <dt class="text-zinc-500">Current label</dt>
  <dd class="text-zinc-200">
    <!-- R6 (visual audit 2026-09-24): a class-less item used to
         read "Current label (vlm)" with a blank value. -->
    {#if current.class_name}
      {current.class_name}
    {:else}
      <span class="italic text-zinc-500">{NO_CLASS_YET}</span>
    {/if}
    {#if current.label_source}
      <span class="ml-1 text-zinc-500">({current.label_source})</span>
    {/if}
    <ConfirmationChip source={current.class_source} validated={current.class_validated} />
    <DetectorClass crop={current} class="ml-1 text-[11px] text-zinc-500" />
    <!-- dq-queues cutover (2026-09-24): vlm_raw_class/
         vlm_class_empty_reason fold into this row rather than
         their own — the review panel is height-budgeted
         (DQ-M5, max-h-[46%] crop cap so the action buttons
         stay visible without scrolling at 1280x720), and every
         extra dt/dd row eats into that budget. -->
    {#if current.vlm_raw_class}
      <span class="ml-1 text-[11px] text-zinc-500"
        >— VLM said: {current.vlm_raw_class}</span
      >
    {:else if current.vlm_class_empty_reason}
      <span class="ml-1 text-[11px] text-orange-300"
        >— {vlmEmptyReasonText(current.vlm_class_empty_reason)}</span
      >
    {/if}
  </dd>

  <!-- m1 (2026-09-24 interactive pass): a name-only proposal
       (proposed_class_id absent — proposed_class_name a
       non-registry term) used to render in
       the same confirmable
       yellow style as a real proposal, though Enter opens the
       picker instead of confirming it — the row now says so. -->
  <dt class="text-zinc-500">Proposed</dt>
  {#if current.proposed_class_name && current.proposed_class_id == null}
    <dd class="text-zinc-400" title="Not a registry class — Enter opens the picker">
      {current.proposed_class_name} <span class="text-[10px]">(hint only)</span>
    </dd>
  {:else}
    <dd class="text-yellow-200">{current.proposed_class_name ?? '—'}</dd>
  {/if}

  {#if opinion.kind === 'no_opinion'}
    <!-- F8 D1: out of the probe's classes: no opinion, never
         shown as a prediction or as agreement. -->
    <dt class="text-zinc-500">Model predicts</dt>
    <dd class="text-zinc-400" data-testid="probe-no-opinion">
      {NO_OPINION_TEXT}
    </dd>
  {:else if opinion.kind === 'unsure'}
    <!-- OpenProcessor 8990ede: a served disagreement below the
         server's own confidence threshold (`probe_actionable`
         false) — shown, but never offered as an Accept action. -->
    <dt class="text-zinc-500">Model predicts</dt>
    <dd class="text-zinc-400" data-testid="probe-unsure">
      model unsure: {current.probe_pred_class}
    </dd>
  {:else if opinion.kind === 'prediction'}
    <!-- G4 closed 2026-09-24 (logic-moves item 14): the backend
       now serves `probe_pred_class_id` alongside the display
       name, so "Accept" no longer needs a client-side
       name→id lookup — assign() takes the served id directly. -->
    <dt class="text-zinc-500">Model predicts</dt>
    <dd class="flex flex-wrap items-center gap-1.5 text-zinc-200">
      {current.probe_pred_class}
      {#if current.probe_pred_entropy != null}
        <ScoreChip label="entropy" value={current.probe_pred_entropy} size="sm" />
      {/if}
      {#if opinion.showAccept}
        <button
          type="button"
          class="rounded border border-blue-500/60 bg-blue-500/15 px-1.5 py-0.5 text-[11px] text-blue-100 hover:bg-blue-500/25"
          onclick={acceptModelClass}
        >
          Accept model's class
        </button>
      {/if}
    </dd>
  {/if}

  {#if current.needs_new_class}
    <dt class="text-zinc-500">Needs new class</dt>
    <dd class="text-amber-200">
      {current.needs_new_class_note || 'flagged — no matching class in the registry'}
    </dd>
  {/if}

  <!-- DQ-M8 (docs/design/data-quality-pass-2026-09-24.md):
       `label_confidence` (wire `confidence`) is the classifier-
       detector score on every row, including VLM-sourced
       ones — the repro was exactly this panel, "Current label
       widget_a (vlm)" directly above "Confidence 94.6%", which
       reads as the VLM's own certainty. Label it for what it is
       whenever the current label came from the VLM (served
       role, not a hardcoded string match), and show the VLM's
       own categorical confidence (`vlm_confidence`, served
       separately) as its own row when there's a real number to
       contrast it with. -->
  <dt class="text-zinc-500">
    {isCurrentLabelVlmSourced ? 'Detector score' : 'Confidence'}
  </dt>
  <dd class="font-mono">
    {current.label_confidence != null
      ? `${(current.label_confidence * 100).toFixed(1)}%`
      : '—'}
    <!-- dq-queues cutover (2026-09-24): class_confidence (the
         confidence of whoever set the CLASS label — distinct
         from label_confidence above, always the detector
         score) folds in here rather than its own row, only
         when there's no VLM-confidence row below to carry it
         instead — see that row's comment for why (DQ-M5
         height budget). -->
    {#if current.class_confidence != null && !current.vlm_confidence}
      <span class="ml-1 text-[11px] text-zinc-500">
        · label: {current.class_confidence_source === 'model' ? 'Model ' : ''}{(
          current.class_confidence * 100
        ).toFixed(0)}%
      </span>
    {/if}
  </dd>

  {#if current.vlm_confidence}
    <dt class="text-zinc-500">VLM confidence</dt>
    <dd class="font-mono text-zinc-200">
      {current.vlm_confidence}
      <!-- class_confidence is the numeric mapping of this same
           categorical value when VLM-sourced — shown inline
           rather than its own row (DQ-M5 height budget, see
           the Detector score/Confidence row's comment above). -->
      {#if current.class_confidence != null}
        <span class="text-[11px] text-zinc-500"
          >({(current.class_confidence * 100).toFixed(0)}%)</span
        >
      {/if}
    </dd>
  {/if}

  {#if current.proposal_name}
    <dt class="text-zinc-500">Proposal hint</dt>
    <dd>
      <span
        class="rounded border border-cyan-500/40 bg-cyan-500/15 px-1.5 py-0.5 text-[11px] text-cyan-200"
        title="A general-purpose detector found a subject here that the primary classifier missed. Coarse class — pick the specific class below (a close variant may be near one-click)."
      >
        {current.proposal_name}
      </span>
    </dd>
  {/if}

  {#if current.crop_rank_in_image != null || current.blur_lap_ratio != null}
    <dt class="text-zinc-500">Rank · clarity</dt>
    <dd class="font-mono text-zinc-300">
      {current.crop_rank_in_image != null
        ? current.crop_rank_in_image === 1
          ? '★1 largest'
          : `#${current.crop_rank_in_image}`
        : '—'}
      {#if current.blur_lap_ratio != null}
        · b{current.blur_lap_ratio.toFixed(2)}
      {/if}
    </dd>
  {/if}

  {#if current.mistakenness_score != null || (current && searchScores.has(current.id))}
    <dt class="text-zinc-500">Scores</dt>
    <dd class="flex flex-wrap items-center gap-1.5">
      {#if current.mistakenness_score != null}
        <ScoreChip
          label="mistakenness"
          value={current.mistakenness_score}
          method={current.mistakenness_method}
          version={current.mistakenness_version}
          size="sm"
        />
      {/if}
      {#if current && searchScores.has(current.id)}
        <ScoreChip label="match" value={searchScores.get(current.id) ?? 0} size="sm" />
      {/if}
    </dd>
  {/if}
</dl>
