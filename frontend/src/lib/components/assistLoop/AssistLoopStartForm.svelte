<script lang="ts">
  /**
   * The loop's request and the server's dry run for it: the plan (clusters,
   * eligible items, the estimated cost of the root level, whether it fits the
   * budget) or the refusal, verbatim. Start is a separate, explicit step.
   */
  import { untrack } from 'svelte';
  import VlmRunPicker from '$components/vlm/VlmRunPicker.svelte';
  import type { AssistLoopController } from '$lib/assistLoop/assistLoopController.svelte';
  import { formatCount } from '$lib/format/count';

  interface Props {
    ctl: AssistLoopController;
  }

  let { ctl }: Props = $props();

  // Re-ask the server for a dry run whenever the form changes.
  $effect(() => {
    void ctl.requestKey;
    untrack(() => ctl.schedulePlan());
  });

  function num(e: Event): number | null {
    const v = (e.currentTarget as HTMLInputElement).valueAsNumber;
    return Number.isNaN(v) ? null : v;
  }

  const plan = $derived(ctl.preview?.plan ?? null);
</script>

<form
  class="flex flex-col gap-4"
  aria-label="Start an auto-label loop"
  data-testid="loop-form"
  onsubmit={(e) => {
    e.preventDefault();
    void ctl.start();
  }}
>
  <div class="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
    <label class="flex flex-col gap-1 text-xs text-zinc-300">
      VLM crop budget
      <input
        class="input"
        type="number"
        min="1"
        required
        value={ctl.maxVlmCrops ?? ''}
        oninput={(e) => (ctl.maxVlmCrops = num(e))}
        data-testid="loop-budget"
      />
    </label>
    <label class="flex flex-col gap-1 text-xs text-zinc-300">
      Smallest cluster to work on
      <input
        class="input"
        type="number"
        min="1"
        value={ctl.minRootSize}
        oninput={(e) => (ctl.minRootSize = num(e) ?? ctl.minRootSize)}
        data-testid="loop-min-root"
      />
    </label>
    <label class="flex flex-col gap-1 text-xs text-zinc-300">
      Homogeneity to show (theta)
      <input
        class="input"
        type="number"
        min="0.7"
        max="0.99"
        step="0.01"
        value={ctl.theta}
        oninput={(e) => (ctl.theta = num(e) ?? ctl.theta)}
        data-testid="loop-theta"
      />
    </label>
    <label class="flex flex-col gap-1 text-xs text-zinc-300">
      Split depth
      <input
        class="input"
        type="number"
        min="0"
        max="6"
        value={ctl.maxDepth}
        oninput={(e) => (ctl.maxDepth = num(e) ?? ctl.maxDepth)}
        data-testid="loop-depth"
      />
    </label>
    <label class="flex flex-col gap-1 text-xs text-zinc-300">
      Seed
      <input
        class="input"
        type="number"
        value={ctl.seed}
        oninput={(e) => (ctl.seed = num(e) ?? 0)}
        data-testid="loop-seed"
      />
    </label>
  </div>

  <div class="flex flex-wrap items-start gap-x-6 gap-y-3 text-xs">
    <VlmRunPicker
      vlm={ctl.vlm}
      acknowledgeExternal={ctl.acknowledgeExternal}
      onchange={(n) => {
        ctl.vlm = n.vlm;
        ctl.acknowledgeExternal = n.acknowledgeExternal;
      }}
    />
    <fieldset class="flex flex-wrap items-center gap-2">
      <legend class="sr-only">Second opinion on what the loop propagates</legend>
      <VlmRunPicker
        vlm={ctl.auditVlm}
        acknowledgeExternal={ctl.auditAcknowledgeExternal}
        defaultLabel="No second opinion"
        onchange={(n) => {
          ctl.auditVlm = n.vlm;
          ctl.auditAcknowledgeExternal = n.acknowledgeExternal;
        }}
      />
      {#if ctl.auditVlm}
        <label class="flex items-center gap-1.5 text-zinc-300">
          If the audit fails
          <select
            class="select-sm"
            value={ctl.auditOnFail}
            onchange={(e) =>
              (ctl.auditOnFail = (e.currentTarget as HTMLSelectElement).value as
                'pause' | 'rollback')}
            data-testid="loop-on-fail"
          >
            <option value="pause">Pause the loop</option>
            <option value="rollback">Roll back the failing batch</option>
          </select>
        </label>
      {/if}
    </fieldset>
  </div>

  <div
    class="rounded border border-zinc-800 bg-zinc-900/40 p-3 text-xs"
    aria-live="polite"
    data-testid="loop-plan"
  >
    <h3 class="mb-1 text-sm font-medium text-zinc-200">Dry run</h3>
    {#if ctl.planError}
      <p class="text-red-300" data-testid="loop-plan-error">{ctl.planError}</p>
    {:else if plan}
      <dl class="grid grid-cols-2 gap-x-6 gap-y-1 sm:grid-cols-4">
        <div>
          <dt class="text-zinc-500">Clusters to work on</dt>
          <dd class="font-mono text-zinc-100" data-testid="loop-plan-roots">
            {formatCount(plan.roots)}
          </dd>
        </div>
        <div>
          <dt class="text-zinc-500">Items it could label</dt>
          <dd class="font-mono text-zinc-100" data-testid="loop-plan-items">
            {formatCount(plan.eligible_items)}
          </dd>
        </div>
        <div>
          <dt class="text-zinc-500">Root level costs about</dt>
          <dd class="font-mono text-zinc-100" data-testid="loop-plan-crops">
            {formatCount(plan.estimated_vlm_crops_root_level)} crops,
            {formatCount(plan.estimated_vlm_requests_root_level)} requests
          </dd>
        </div>
        <div>
          <dt class="text-zinc-500">Fits the budget</dt>
          <dd class="font-mono text-zinc-100" data-testid="loop-plan-fits">
            {plan.fits_budget === null ? '—' : plan.fits_budget ? 'yes' : 'no'}
          </dd>
        </div>
      </dl>
      <p class="mt-2 text-zinc-500">
        Splits cost more than the root-level estimate; the budget is a hard cap and the
        loop stops at it.
        {#if ctl.preview?.pinned.audit_vlm === null}
          No second opinion: the labels it propagates are not re-checked.
        {/if}
      </p>
      {#if plan.required_sample_table.length > 0}
        <details class="mt-2">
          <summary class="cursor-pointer text-zinc-300"
            >Answers needed per cluster size</summary
          >
          <table class="mt-1 w-full text-left" data-testid="loop-sample-table">
            <caption class="sr-only"
              >VLM answers a cluster of each size needs to pass, by sampled answers
              allowed to disagree</caption
            >
            <thead class="text-zinc-500">
              <tr>
                <th scope="col" class="pr-4 font-normal">Cluster size</th>
                <th scope="col" class="font-normal">Answers at 0, 1, 2, 3 dissent</th>
              </tr>
            </thead>
            <tbody class="font-mono text-zinc-200">
              {#each plan.required_sample_table as row (row.n_members)}
                <tr>
                  <td class="pr-4">{formatCount(row.n_members)}</td>
                  <td>{row.required_by_dissent.join(', ')}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </details>
      {/if}
    {:else}
      <p class="text-zinc-500">{ctl.planning ? 'Planning…' : 'No plan yet.'}</p>
    {/if}
  </div>

  {#if ctl.actionError}
    <p class="text-xs text-red-300" role="alert" data-testid="loop-action-error">
      {ctl.actionError}
    </p>
  {/if}

  <div class="flex items-center gap-3">
    <button
      type="submit"
      class="btn btn-primary btn-sm"
      disabled={!ctl.canStart}
      data-testid="loop-start"
    >
      Start the loop
    </button>
    <span class="text-xs text-zinc-500">
      Writes machine labels (not validated, kept out of the export until reviewed). A
      rollback restores them.
    </span>
  </div>
</form>
