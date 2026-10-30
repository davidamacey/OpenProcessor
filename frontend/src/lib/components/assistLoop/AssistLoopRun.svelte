<script lang="ts">
  /**
   * A loop in flight or finished: its status and why it stopped, the budget
   * pools against their caps, per-level counts, the totals, the audit result and
   * the controls the server offers (cancel while live, resume when stopped).
   * Every figure is the server's.
   */
  import type { AssistLoopController } from '$lib/assistLoop/assistLoopController.svelte';
  import { isResumableLoop } from '$lib/assistLoop/assistLoopController.svelte';
  import { formatCount } from '$lib/format/count';
  import { formatDuration } from '$lib/format/duration';
  import { formatTimestamp } from '$lib/format/date';
  import type { LoopPoolBudget } from '$lib/types_assistLoop';

  interface Props {
    ctl: AssistLoopController;
  }

  let { ctl }: Props = $props();

  const loop = $derived(ctl.loop);

  const STATUS_TEXT: Record<string, string> = {
    queued: 'Queued for the worker',
    running: 'Running',
    paused: 'Paused',
    completed: 'Completed',
    cancelled: 'Cancelled',
    failed: 'Failed',
    interrupted: 'Interrupted (the worker stopped); resume to continue',
  };
  const STOP_TEXT: Record<string, string> = {
    done: 'every cluster was decided',
    budget_crops: 'the crop budget was spent',
    budget_requests: 'the request budget was spent',
    budget_time: 'the time budget ran out',
  };
  const PAUSE_TEXT: Record<string, string> = {
    user: 'paused by a person',
    audit: 'the second opinion disagreed with too many labels',
  };

  function pct(spent: number, max: number): number {
    return max > 0 ? Math.min(100, (spent / max) * 100) : 0;
  }
</script>

{#snippet pool(name: string, b: LoopPoolBudget)}
  <div class="flex flex-col gap-1" data-testid="loop-pool-{name.toLowerCase()}">
    <div class="flex justify-between gap-2">
      <span class="text-zinc-400">{name} crops</span>
      <span class="font-mono text-zinc-100"
        >{formatCount(b.crops_spent)} / {formatCount(b.max_crops)}</span
      >
    </div>
    <div
      class="h-1.5 rounded bg-zinc-800"
      role="progressbar"
      aria-label="{name} crops spent"
      aria-valuemin="0"
      aria-valuemax={b.max_crops}
      aria-valuenow={b.crops_spent}
    >
      <div
        class="h-1.5 rounded bg-sky-500/70"
        style="width: {pct(b.crops_spent, b.max_crops)}%"
      ></div>
    </div>
    <div class="flex justify-between gap-2 text-zinc-500">
      <span>{formatCount(b.crops_reserved)} in flight</span>
      <span>{formatCount(b.requests_spent)} / {formatCount(b.max_requests)} requests</span
      >
    </div>
  </div>
{/snippet}

{#if loop}
  <div
    class="flex flex-col gap-4 text-xs"
    data-testid="loop-run"
    data-status={loop.status}
  >
    <div class="flex flex-wrap items-center gap-3">
      <p class="text-sm text-zinc-100" role="status" data-testid="loop-status">
        {STATUS_TEXT[loop.status] ?? loop.status}
        {#if loop.status === 'paused' && loop.pause_reason}
          <span class="text-zinc-400"
            >: {PAUSE_TEXT[loop.pause_reason] ?? loop.pause_reason}</span
          >
        {:else if loop.stop_reason}
          <span class="text-zinc-400"
            >: {STOP_TEXT[loop.stop_reason] ?? loop.stop_reason}</span
          >
        {/if}
      </p>
      <span class="font-mono text-zinc-500" title={loop.created_at ?? ''}
        >{loop.loop_id} · {formatTimestamp(loop.started_at ?? loop.created_at)}</span
      >
      <div class="ml-auto flex gap-2">
        {#if ctl.live}
          <button
            type="button"
            class="btn btn-sm"
            disabled={ctl.busy}
            onclick={() => void ctl.cancel()}
            data-testid="loop-cancel"
          >
            Cancel the loop
          </button>
        {/if}
        {#if isResumableLoop(loop.status)}
          <button
            type="button"
            class="btn btn-sm btn-primary"
            disabled={ctl.busy}
            onclick={() => void ctl.resume()}
            data-testid="loop-resume"
          >
            Resume the loop
          </button>
        {/if}
      </div>
    </div>

    {#if loop.error}
      <p class="text-red-300" data-testid="loop-error">{loop.error}</p>
    {/if}
    {#if ctl.actionError}
      <p class="text-red-300" role="alert" data-testid="loop-action-error">
        {ctl.actionError}
      </p>
    {/if}

    {#if loop.budget}
      <div class="grid grid-cols-1 gap-4 sm:grid-cols-3" data-testid="loop-budget-view">
        {@render pool('VLM', loop.budget.vlm)}
        {#if loop.budget.audit}
          {@render pool('Audit', loop.budget.audit)}
        {/if}
        <div class="flex flex-col gap-1">
          <span class="text-zinc-400">Time</span>
          <span class="font-mono text-zinc-100" data-testid="loop-wall"
            >{formatDuration(loop.budget.wall_s)} used</span
          >
        </div>
      </div>
    {/if}

    {#if loop.totals}
      <dl
        class="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-4"
        data-testid="loop-totals"
      >
        <div>
          <dt class="text-zinc-500">Labels propagated</dt>
          <dd class="font-mono text-base text-zinc-100" data-testid="loop-propagated">
            {formatCount(loop.totals.items_propagated)}
          </dd>
        </div>
        <div>
          <dt class="text-zinc-500">Labeled by the VLM directly</dt>
          <dd class="font-mono text-base text-zinc-100">
            {formatCount(loop.totals.items_direct)}
          </dd>
        </div>
        <div>
          <dt class="text-zinc-500">Clusters left for review</dt>
          <dd class="font-mono text-base text-zinc-100">
            {formatCount(loop.totals.clusters_left_for_review)}
          </dd>
        </div>
        <div>
          <dt class="text-zinc-500">Items left for review</dt>
          <dd class="font-mono text-base text-zinc-100">
            {formatCount(loop.totals.items_left_for_review)}
          </dd>
        </div>
      </dl>
      {#if Object.values(loop.totals.skipped).some((n) => n > 0)}
        <p class="text-zinc-500" data-testid="loop-skipped">
          Skipped:
          {Object.entries(loop.totals.skipped)
            .filter(([, n]) => n > 0)
            .map(([k, n]) => `${formatCount(n)} ${k.replace('_', ' ')}`)
            .join(', ')}.
        </p>
      {/if}
    {/if}

    {#if loop.levels.length > 0}
      <table class="w-full text-left" data-testid="loop-levels">
        <caption class="mb-1 text-left text-sm font-medium text-zinc-200"
          >Progress by level</caption
        >
        <thead class="text-zinc-500">
          <tr>
            <th scope="col" class="pr-3 font-normal">Level</th>
            <th scope="col" class="pr-3 font-normal">Clusters</th>
            <th scope="col" class="pr-3 font-normal">Passed</th>
            <th scope="col" class="pr-3 font-normal">Split</th>
            <th scope="col" class="pr-3 font-normal">For review</th>
            <th scope="col" class="pr-3 font-normal">Stale</th>
            <th scope="col" class="pr-3 font-normal">Not visited</th>
            <th scope="col" class="pr-3 font-normal">Labels propagated</th>
            <th scope="col" class="font-normal">Crops spent</th>
          </tr>
        </thead>
        <tbody class="font-mono text-zinc-200">
          {#each loop.levels as l (l.level)}
            <tr>
              <th scope="row" class="pr-3 text-left font-normal">{l.level}</th>
              <td class="pr-3">{formatCount(l.nodes)}</td>
              <td class="pr-3">{formatCount(l.passed)}</td>
              <td class="pr-3">{formatCount(l.split)}</td>
              <td class="pr-3">{formatCount(l.review_leaf + l.review_small)}</td>
              <td class="pr-3">{formatCount(l.stale)}</td>
              <td class="pr-3">{formatCount(l.unvisited)}</td>
              <td class="pr-3">{formatCount(l.items_propagated)}</td>
              <td>{formatCount(l.crops_spent)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}

    {#if loop.audit}
      <div data-testid="loop-audit" data-audit-status={loop.audit.status}>
        <h3 class="mb-1 text-sm font-medium text-zinc-200">Second opinion</h3>
        {#if loop.audit.status === 'off'}
          <p class="text-zinc-400">
            Off: no second model re-checked the labels this loop propagated.
          </p>
        {:else}
          <p class="text-zinc-300">
            <span data-testid="loop-audit-status"
              >{loop.audit.status.replace('_', ' ')}</span
            >:
            {formatCount(loop.audit.agreed)} of {formatCount(loop.audit.sampled)} sampled labels
            agreed
            {#if loop.audit.agreement !== null}
              ({(loop.audit.agreement * 100).toFixed(1)} percent{#if loop.audit.lower_bound !== null},
                lower bound {(loop.audit.lower_bound * 100).toFixed(1)} percent{/if}){/if}.
            {formatCount(loop.audit.crops_spent)} audit crops spent.
          </p>
        {/if}
      </div>
    {/if}
  </div>
{/if}
