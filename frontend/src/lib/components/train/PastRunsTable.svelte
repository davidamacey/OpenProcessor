<script lang="ts">
  import Icon from '$components/ui/Icon.svelte';
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

<section class="rounded-md border border-line bg-surface-1">
  <header class="flex items-center justify-between gap-3 border-b border-line px-3 py-2">
    <h2 class="text-body font-semibold text-fg-1">Past runs</h2>
    <span class="font-mono text-meta text-fg-3">{ctl.runs.length} / {ctl.runsTotal}</span>
  </header>
  {#if ctl.runsError}
    <p class="px-3 py-2 text-meta text-danger-text">
      Past runs fetch failed: {ctl.runsError}
    </p>
  {/if}
  <div class="max-h-[40rem] overflow-auto">
    {#if ctl.runsLoading && ctl.runs.length === 0}
      <p class="px-3 py-3 text-meta text-fg-3">Loading…</p>
    {:else if ctl.runs.length === 0}
      <p class="px-3 py-3 text-meta text-fg-3">No runs yet. Start one above.</p>
    {:else}
      <table class="w-full min-w-[42rem] table-fixed text-body">
        <thead
          class="sticky top-0 bg-surface-1 text-caption uppercase tracking-wide text-fg-3"
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
            <tr class="border-t border-line hover:bg-surface-2/50">
              <td class="px-3 py-2">
                <div class="truncate font-mono text-meta text-fg-1" title={r.job_id}>
                  {r.job_id}
                </div>
                {#if r.campaign_id}
                  <div
                    class="truncate font-mono text-caption text-fg-3"
                    title={r.campaign_id}
                  >
                    <Icon
                      name="corner-down-right"
                      size={12}
                      class="mr-1 -mt-0.5 inline-block"
                    />{r.campaign_id}
                  </div>
                {/if}
              </td>
              <td class="px-3 py-2">
                <span class="font-mono text-meta text-fg-2">{familyLabel(r)}</span>
              </td>
              <td class="px-3 py-2">
                <span
                  class="rounded-sm border px-1.5 py-0.5 text-caption font-medium uppercase tracking-wide {statePillClass(
                    r.state,
                  )}"
                >
                  {r.state}
                </span>
              </td>
              <td
                class="px-3 py-2 text-right font-mono text-meta text-fg-1"
                title={mapDisplay.source === 'test'
                  ? 'test split (frozen holdout)'
                  : mapDisplay.source === 'val'
                    ? 'validation (test pass fell back)'
                    : 'no eval served yet'}
              >
                {mapDisplay.value?.toFixed(3) ?? '—'}
                <span class="block font-sans text-caption text-fg-3">
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
                      class="rounded border border-line-strong bg-surface-0 px-2 py-1 text-meta text-accent-text hover:border-accent hover:bg-accent/10"
                      onclick={() => actions.openPromote(r)}
                      title="Promote to Triton"
                    >
                      Promote<Icon name="arrow-up" size={14} class="ml-1 inline-block" />
                    </button>
                  {/if}
                  {#if r.state === 'finished' || r.state === 'failed'}
                    <button
                      type="button"
                      class="rounded border border-line-strong bg-surface-0 px-2 py-1 text-meta text-fg-2 hover:border-fg-3 hover:bg-surface-2"
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
