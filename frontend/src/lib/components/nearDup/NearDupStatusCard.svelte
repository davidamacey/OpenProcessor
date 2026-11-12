<!--
  Per level: the current run, what is pending and why the groups are stale, and
  the run controls. Every figure is served by `GET /near_duplicates/status`.
-->
<script lang="ts">
  import { formatTimestamp } from '$lib/format/date';
  import { runModeFor, staleReasonText } from '$lib/nearDup/staleRefusal';
  import type { NearDupPageController } from '$lib/nearDup/nearDupPageController.svelte';
  import type { NearDupLevel, NearDupLevelStatus } from '$lib/types_nearDup';

  let { ctl }: { ctl: NearDupPageController } = $props();

  const LABEL: Record<NearDupLevel, string> = {
    image: 'Image level (whole frames)',
    item: 'Item level (crops inside leak groups)',
  };

  function pendingOnly(lv: NearDupLevelStatus): boolean {
    return runModeFor(lv) === 'incremental';
  }
</script>

<section class="surface flex flex-col gap-3 p-4" data-testid="near-dup-status">
  <header class="flex flex-wrap items-baseline gap-3">
    <h2 class="text-title font-semibold">Status</h2>
    {#if ctl.status?.profile}
      <span class="text-meta text-fg-2" data-testid="near-dup-profile-chip">
        Profile <span class="font-mono text-fg-1">{ctl.status.profile.name}</span>
        revision {ctl.status.profile.revision}
      </span>
    {:else if ctl.status}
      <span class="text-meta text-warn-text">No active profile: nothing groups.</span>
    {/if}
  </header>

  {#if ctl.runMessage}
    <p class="text-meta text-fg-2" role="status" data-testid="near-dup-run-message">
      {ctl.runMessage}
    </p>
  {/if}
  {#if ctl.runError}
    <p class="text-meta text-danger-text" role="alert" data-testid="near-dup-run-error">
      {ctl.runError}
    </p>
  {/if}

  <div class="grid gap-3 md:grid-cols-2">
    {#each ctl.status?.levels ?? [] as lv (lv.level)}
      <div
        class="flex flex-col gap-2 rounded border border-line p-3"
        data-testid="near-dup-level-{lv.level}"
      >
        <h3 class="text-body font-medium">{LABEL[lv.level]}</h3>
        {#if lv.current_run}
          <p class="text-meta text-fg-2">
            Current run <span class="font-mono">{lv.current_run.mode}</span>,
            {formatTimestamp(lv.current_run.finished_at ?? lv.current_run.started_at)}:
            {lv.current_run.dedup_groups} duplicate groups,
            {lv.current_run.leak_groups} leak groups,
            {lv.current_run.grouped_members} of {lv.current_run.members_scored} members grouped.
          </p>
        {:else}
          <p class="text-meta text-fg-2" data-testid="near-dup-no-run-{lv.level}">
            No run yet. Start a full build to group this level.
          </p>
        {/if}
        {#if lv.stale_reasons.length > 0}
          <ul class="text-meta text-warn-text" data-testid="near-dup-stale-{lv.level}">
            {#each lv.stale_reasons as reason (reason)}
              <li>
                Stale: {staleReasonText(reason)}{reason === 'pending'
                  ? ` (${lv.pending_count})`
                  : ''}
              </li>
            {/each}
          </ul>
        {:else}
          <p class="text-meta text-success-text">Up to date.</p>
        {/if}
        <div class="mt-auto flex flex-wrap gap-2">
          {#if lv.pending_count > 0 && lv.current_run && pendingOnly(lv)}
            <button
              type="button"
              class="btn btn-primary"
              disabled={ctl.busyLevel !== null}
              data-testid="near-dup-run-incremental-{lv.level}"
              onclick={() => void ctl.startRun(lv.level, 'incremental')}
              >Group new items</button
            >
          {/if}
          <button
            type="button"
            class="btn"
            disabled={ctl.busyLevel !== null}
            data-testid="near-dup-run-full-{lv.level}"
            onclick={() => void ctl.startRun(lv.level, 'full')}
            >{ctl.busyLevel === lv.level
              ? 'Running…'
              : lv.current_run
                ? 'Full rebuild'
                : 'Build'}</button
          >
        </div>
      </div>
    {/each}
  </div>
</section>
