<script lang="ts">
  /** /train's paginated past-runs table with per-run promote/reproduce. */
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import RunResults from '$components/RunResults.svelte';
  import { bestMapDisplay } from '$lib/trainRunsTable';
  import { isTerminalTrainState } from '$lib/trainResults';
  import { familyLabel, statePillClass } from '$lib/train/trainDisplay';
  import type { TrainRunsController } from '$lib/train/trainRunsController.svelte';
  import type { TrainActionsController } from '$lib/train/trainActionsController.svelte';

  interface Props {
    ctl: TrainRunsController;
    actions: TrainActionsController;
  }

  let { ctl, actions }: Props = $props();
</script>

<section class="rounded-md border border-zinc-800 bg-zinc-900">
  <header
    class="flex items-center justify-between gap-3 border-b border-zinc-800 px-3 py-2"
  >
    <h2 class="text-sm font-semibold text-zinc-100">Past runs</h2>
    <span class="font-mono text-xs text-zinc-500"
      >{ctl.runs.length} / {ctl.runsTotal}</span
    >
  </header>
  {#if ctl.runsError}
    <p class="px-3 py-2 text-xs text-red-300">Past runs fetch failed: {ctl.runsError}</p>
  {/if}
  <div class="max-h-[40rem] overflow-auto">
    {#if ctl.runsLoading && ctl.runs.length === 0}
      <p class="px-3 py-3 text-xs text-zinc-500">Loading…</p>
    {:else if ctl.runs.length === 0}
      <p class="px-3 py-3 text-xs text-zinc-500">No runs yet. Start one above.</p>
    {:else}
      <table class="w-full table-fixed text-sm">
        <thead
          class="sticky top-0 bg-zinc-900 text-[11px] uppercase tracking-wide text-zinc-500"
        >
          <tr>
            <th class="px-3 py-2 text-left">Name</th>
            <th class="w-20 px-3 py-2 text-left">Family</th>
            <th class="w-24 px-3 py-2 text-left">Status</th>
            <th
              class="w-28 px-3 py-2 text-right"
              title="The run's served eval.map50, labelled by eval.split (test split when the frozen-holdout pass ran, val when it fell back)"
            >
              mAP50
            </th>
            <th class="w-48 px-3 py-2 text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          {#each ctl.runs as r (r.job_id)}
            {@const mapDisplay = bestMapDisplay(r)}
            <tr class="border-t border-zinc-800 hover:bg-zinc-800/50">
              <td class="px-3 py-2">
                <div class="truncate font-mono text-xs text-zinc-200" title={r.job_id}>
                  {r.job_id}
                </div>
                {#if r.campaign_id}
                  <div
                    class="truncate font-mono text-[10px] text-zinc-500"
                    title={r.campaign_id}
                  >
                    ↳ {r.campaign_id}
                  </div>
                {/if}
              </td>
              <td class="px-3 py-2">
                <span class="font-mono text-xs text-zinc-400">{familyLabel(r)}</span>
              </td>
              <td class="px-3 py-2">
                <span
                  class="rounded-sm border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide {statePillClass(
                    r.state,
                  )}"
                >
                  {r.state}
                </span>
              </td>
              <td
                class="px-3 py-2 text-right font-mono text-xs text-zinc-200"
                title={mapDisplay.source === 'test'
                  ? 'test split (frozen holdout)'
                  : mapDisplay.source === 'val'
                    ? 'validation (test pass fell back)'
                    : 'no eval served yet'}
              >
                {mapDisplay.value?.toFixed(3) ?? '—'}
                <span class="block font-sans text-[9px] text-zinc-500">
                  {mapDisplay.source === 'test'
                    ? 'test'
                    : mapDisplay.source === 'val'
                      ? 'val'
                      : '—'}
                </span>
              </td>
              <td class="px-3 py-2">
                <div class="flex flex-wrap justify-end gap-1.5">
                  {#if r.state === 'finished' || r.state === 'exporting'}
                    <button
                      type="button"
                      class="rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-blue-300 hover:border-blue-500 hover:bg-blue-500/10"
                      onclick={() => actions.openPromote(r)}
                      title="Promote to Triton"
                    >
                      Promote ↑
                    </button>
                  {/if}
                  {#if r.state === 'finished' || r.state === 'failed'}
                    <button
                      type="button"
                      class="rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-300 hover:border-zinc-500 hover:bg-zinc-800"
                      onclick={() => actions.reproduceRun(r)}
                      disabled={actions.reproducingId === r.job_id}
                      title="Submit a new run with the same spec"
                    >
                      {actions.reproducingId === r.job_id ? '…' : 'Reproduce'}
                    </button>
                  {/if}
                </div>
              </td>
            </tr>
            {#if isTerminalTrainState(r.state)}
              <tr class="border-t-0">
                <td colspan="5" class="p-0">
                  <RunResults status={r} />
                </td>
              </tr>
            {/if}
          {/each}
        </tbody>
      </table>
      <div
        use:infiniteScroll={{
          onload: ctl.loadMoreRuns,
          disabled: ctl.runsLoadingMore || !ctl.runsHasMore || ctl.runsLoading,
        }}
        class="h-1"
        aria-hidden="true"
      ></div>
    {/if}
  </div>
</section>
