<!--
  The run history, newest first, as served by `GET /near_duplicates/runs`.
-->
<script lang="ts">
  import { formatTimestamp } from '$lib/format/date';
  import type { NearDupRunView } from '$lib/types_nearDup';

  let { runs }: { runs: NearDupRunView[] } = $props();
</script>

<section class="surface flex flex-col gap-2 p-4" data-testid="near-dup-runs">
  <h2 class="text-base font-semibold">Run history</h2>
  {#if runs.length === 0}
    <p class="text-xs text-fg-2" data-testid="near-dup-runs-empty">No runs yet.</p>
  {:else}
    <div class="overflow-x-auto">
      <table class="w-full text-left text-xs">
        <thead class="text-fg-2">
          <tr>
            <th class="py-1 pr-3 font-medium">Started</th>
            <th class="pr-3 font-medium">Level</th>
            <th class="pr-3 font-medium">Mode</th>
            <th class="pr-3 font-medium">Status</th>
            <th class="pr-3 font-medium">Profile rev.</th>
            <th class="pr-3 font-medium">Scored</th>
            <th class="pr-3 font-medium">Dup groups</th>
            <th class="pr-3 font-medium">Leak groups</th>
            <th class="pr-3 font-medium">Largest</th>
            <th class="pr-3 font-medium">Split</th>
            <th class="font-medium">Holdout absorbed</th>
          </tr>
        </thead>
        <tbody>
          {#each runs as run (run.id)}
            <tr class="border-t border-line" data-testid="near-dup-run-row">
              <td class="py-1 pr-3">{formatTimestamp(run.started_at)}</td>
              <td class="pr-3">{run.level}</td>
              <td class="pr-3">{run.mode}</td>
              <td class="pr-3">{run.status}</td>
              <td class="pr-3">{run.profile_revision ?? '-'}</td>
              <td class="pr-3">{run.members_scored}</td>
              <td class="pr-3">{run.dedup_groups}</td>
              <td class="pr-3">{run.leak_groups}</td>
              <td class="pr-3">{run.largest_group}</td>
              <td class="pr-3">{run.oversized_split}</td>
              <td>{run.holdout_absorbed}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</section>
