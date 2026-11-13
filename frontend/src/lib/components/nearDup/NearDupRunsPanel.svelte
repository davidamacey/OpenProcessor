<!--
  Per level: the current run, what is pending and why the groups are stale, and the full
  rebuild. Every figure is served by `GET /near_duplicates/status`. The quick "group new items"
  run lives where the user sees its result, in the Workspace Gallery.
-->
<script lang="ts">
  import Button from '$components/ui/Button.svelte';
  import { formatTimestamp } from '$lib/format/date';
  import { staleReasonText } from '$lib/nearDup/staleRefusal';
  import type { NearDupPageController } from '$lib/nearDup/nearDupPageController.svelte';
  import type { NearDupLevel } from '$lib/types_nearDup';

  let { ctl }: { ctl: NearDupPageController } = $props();

  const LABEL: Record<NearDupLevel, string> = {
    image: 'Photos (whole frames)',
    item: 'Objects (crops inside leak groups)',
  };
</script>

<div class="flex flex-col gap-3" data-testid="near-dup-status">
  {#if ctl.status && !ctl.status.profile}
    <p class="text-meta text-warn-text">No profile is active: nothing is grouped.</p>
  {/if}
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
        <h3 class="text-body font-medium text-fg-1">{LABEL[lv.level]}</h3>
        {#if lv.current_run}
          <p class="text-meta text-fg-2">
            Last run ({lv.current_run.mode}),
            {formatTimestamp(lv.current_run.finished_at ?? lv.current_run.started_at)}:
            {lv.current_run.dedup_groups} duplicate groups,
            {lv.current_run.leak_groups} leak groups,
            {lv.current_run.grouped_members} of {lv.current_run.members_scored} grouped.
          </p>
        {:else}
          <p class="text-meta text-fg-2" data-testid="near-dup-no-run-{lv.level}">
            Not built yet.
          </p>
        {/if}
        {#if lv.stale_reasons.length > 0}
          <ul class="text-meta text-warn-text" data-testid="near-dup-stale-{lv.level}">
            {#each lv.stale_reasons as reason (reason)}
              <li>
                Out of date: {staleReasonText(reason)}{reason === 'pending'
                  ? ` (${lv.pending_count})`
                  : ''}
              </li>
            {/each}
          </ul>
        {:else}
          <p class="text-meta text-success-text">Up to date.</p>
        {/if}
        <div class="mt-auto">
          <Button
            disabled={ctl.busyLevel !== null}
            data-testid="near-dup-run-full-{lv.level}"
            onclick={() => void ctl.startRun(lv.level, 'full')}
            >{ctl.busyLevel === lv.level
              ? 'Running…'
              : lv.current_run
                ? 'Full rebuild'
                : 'Build'}</Button
          >
        </div>
      </div>
    {/each}
  </div>
</div>
