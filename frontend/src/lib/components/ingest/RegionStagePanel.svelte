<script lang="ts">
  /**
   * The region stage's served state on `/ingest` (`GET /region_stage`):
   * paused or running, since when, whether the whole project pipeline is
   * paused, and the worklog counts. Pause and Resume are confirm-gated;
   * "Re-run gate-skipped" opens the reprocess dialog with the request the
   * server served, as served. Mounted only while a region profile is
   * served, because the routes answer 409 without one.
   */
  import { onMount } from 'svelte';
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import ReprocessControl from '$components/datasets/ReprocessControl.svelte';
  import { createRegionStage } from '$lib/openVocab/regionStageController.svelte';
  import { formatTimestamp } from '$lib/format/date';

  const stage = createRegionStage();

  onMount(() => {
    void stage.load();
  });
</script>

<div class="mt-4 border-t border-line pt-3" data-testid="region-stage-panel">
  <h3 class="mb-2 text-sm font-semibold text-fg-1">Region stage</h3>
  {#if stage.loadError && !stage.state}
    <p class="text-xs text-danger" data-testid="region-stage-error">
      Region stage unavailable: {stage.loadError}
    </p>
  {:else if !stage.state}
    <p class="text-xs text-fg-3">Loading…</p>
  {:else}
    {@const s = stage.state}
    <div class="flex flex-wrap items-center gap-2 text-xs">
      <span
        class="rounded border px-1.5 py-0.5 {s.paused
          ? 'border-warn/40 bg-warn/10 text-warn'
          : 'border-success/40 bg-success/10 text-success'}"
        data-testid="region-stage-state">{s.paused ? 'paused' : 'running'}</span
      >
      {#if s.paused && s.paused_since}
        <span class="text-fg-3" title={s.paused_since} data-testid="region-stage-since"
          >since {formatTimestamp(s.paused_since)}</span
        >
      {/if}
      <span class="grow"></span>
      <button
        type="button"
        class="btn btn-sm"
        disabled={stage.busy}
        data-testid="region-stage-toggle"
        onclick={() => stage.ask(s.paused ? 'resume' : 'pause')}
        >{s.paused ? 'Resume' : 'Pause'}</button
      >
    </div>
    {#if s.pipeline_paused}
      <p class="mt-2 text-xs text-warn" data-testid="region-stage-pipeline-paused">
        This project's whole pipeline is paused, so the region stage is not running either
        way.
      </p>
    {/if}
    <div class="mt-2 flex flex-wrap gap-4 text-xs">
      <div>
        <div class="text-fg-3">Pending detection</div>
        <div class="font-mono text-fg-1" data-testid="region-stage-pending-detection">
          {s.counts.pending_detection}
        </div>
      </div>
      <div>
        <div class="text-fg-3">Pending verification</div>
        <div class="font-mono text-fg-1" data-testid="region-stage-pending-verification">
          {s.counts.pending_verification}
        </div>
      </div>
      <div>
        <div class="text-fg-3">Gate-skipped</div>
        <div class="font-mono text-fg-1" data-testid="region-stage-gate-skipped">
          {s.counts.gate_skipped}
        </div>
      </div>
    </div>
    {#if s.counts.gate_skipped > 0 && stage.rerunRequest}
      <div class="mt-2">
        <ReprocessControl
          target={{ kind: 'request', request: stage.rerunRequest }}
          buttonLabel="Re-run gate-skipped ({s.counts.gate_skipped})…"
        />
      </div>
    {/if}
  {/if}
</div>

{#if stage.confirming}
  <ConfirmDialog
    title={stage.confirming === 'pause'
      ? 'Pause the region stage'
      : 'Resume the region stage'}
    confirmLabel={stage.confirming === 'pause' ? 'Pause' : 'Resume'}
    busy={stage.busy}
    onconfirm={() => void stage.confirm()}
    oncancel={() => stage.cancel()}
  >
    {#if stage.confirming === 'pause'}
      <p>
        Pause the region stage for this project? Queued items stay pending; nothing is
        lost.
      </p>
    {:else}
      <p>Resume the region stage? Queued items are picked up again.</p>
    {/if}
    {#if stage.actionError}<p class="text-danger" data-testid="region-stage-action-error">
        {stage.actionError}
      </p>{/if}
  </ConfirmDialog>
{/if}
