<script lang="ts">
  /**
   * The VLM's proposed label for a cluster or sub-cluster, from one assist
   * run: the class, how strongly the sampled answers agree, the votes and
   * the attribute suggestions. Applying it is the explicit button, shown
   * with the server's dry-run count; the server decides what is applicable,
   * refuses a scope that changed since the run (`scope_changed`) and a class
   * the registry lacks. Attributes are suggestions, never labels.
   */
  import { untrack } from 'svelte';
  import AddClassModal from '$components/AddClassModal.svelte';
  import { ProposalApplyController } from '$lib/assistRun/proposalApplyController.svelte';
  import { scopeLabel } from '$lib/assistRun/scopeLabel';
  import type { AssistRunState } from '$lib/types_assist';

  interface Props {
    run: AssistRunState;
    /** Called after the class was written, so the page can reload. */
    onapplied?: () => void;
  }

  let { run, onapplied }: Props = $props();

  const proposal = $derived(run.result?.proposal ?? null);
  const ctl = $derived(
    new ProposalApplyController(run.run_id ?? '', undefined, () => onapplied?.()),
  );
  let addClassOpen = $state(false);
  // The user's pick; starts as the class that passed the gate, if any.
  let picked = $state<string | null>(null);
  const chosen = $derived(picked ?? proposal?.class_name ?? null);

  $effect(() => {
    const name = chosen;
    const c = ctl;
    untrack(() => {
      c.choose(name);
      if (name) void c.preview();
    });
  });

  const pct = (x: number) => `${Math.round(x * 100)}%`;
  const maxVotes = $derived(
    Math.max(1, ...(proposal?.distribution.map((d) => d.votes) ?? [])),
  );
</script>

{#if proposal && run.run_id}
  <section
    class="flex flex-col gap-3 rounded-md border border-sky-500/30 bg-sky-500/5 p-3 text-sm"
    data-testid="assist-proposal-card"
  >
    <header class="flex flex-wrap items-baseline gap-x-3 gap-y-1">
      <h3 class="text-sm font-semibold text-zinc-100">
        VLM proposal for {scopeLabel(run.request.target)}
      </h3>
      <span class="text-xs text-zinc-500">run {run.run_id}</span>
    </header>

    {#if run.scope_now?.changed}
      <p
        class="rounded border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-xs text-amber-200"
        data-testid="proposal-stale"
      >
        The scope has changed since this run: {run.scope_now.overlap} of the
        {run.scope_now.count} members now were in it ({pct(run.scope_now.jaccard)} overlap),
        for example after a re-refine. Run it again on the current members.
      </p>
    {/if}

    {#if proposal.class_name}
      <p class="flex flex-wrap items-baseline gap-2">
        <span class="text-zinc-400">Proposed class</span>
        <strong class="text-base text-zinc-50" data-testid="proposal-class"
          >{proposal.class_name}</strong
        >
        {#if !proposal.in_registry}
          <span class="text-xs text-amber-300">not in the class registry</span>
        {/if}
      </p>
    {:else}
      <p class="text-zinc-300" data-testid="proposal-nogate">
        No class reached the agreement gate: the top class got {pct(proposal.agreement)} of
        {proposal.answered} answers, and {pct(proposal.min_agreement)} is needed.
      </p>
    {/if}

    <p class="text-xs text-zinc-400" data-testid="proposal-agreement">
      {pct(proposal.agreement)} agreement across {proposal.answered} sampled answers (gate
      {pct(proposal.min_agreement)}), from {run.resolved.sampled} of
      {run.resolved.scope_count} members.
    </p>

    <ul class="flex flex-col gap-1" data-testid="proposal-distribution">
      {#each proposal.distribution as d (d.class_name)}
        <li class="flex items-center gap-2 text-xs">
          <span class="w-32 shrink-0 truncate text-zinc-200" title={d.class_name}
            >{d.class_name}</span
          >
          <span class="h-2 grow rounded bg-zinc-800">
            <span
              class="block h-2 rounded bg-sky-500/60"
              style="width: {(d.votes / maxVotes) * 100}%"
            ></span>
          </span>
          <span class="w-8 shrink-0 text-right font-mono text-zinc-400">{d.votes}</span>
          <button
            type="button"
            class="btn-sm"
            data-testid="proposal-use-{d.class_name}"
            disabled={chosen === d.class_name}
            onclick={() => (picked = d.class_name)}>Use</button
          >
        </li>
      {/each}
    </ul>

    {#if proposal.new_class_proposals.length > 0}
      <p class="text-xs text-zinc-400" data-testid="proposal-new-classes">
        The VLM also proposed classes the registry lacks:
        {#each proposal.new_class_proposals as n, i (n.name)}
          <span class="text-zinc-200">{n.name}</span> ({n.votes}){i <
          proposal.new_class_proposals.length - 1
            ? ', '
            : ''}
        {/each}
      </p>
    {/if}

    {#if proposal.attribute_summary.length > 0}
      <div class="flex flex-col gap-1" data-testid="proposal-attributes">
        <span class="text-[10px] uppercase tracking-wide text-zinc-500"
          >Attribute suggestions, not labels</span
        >
        {#each proposal.attribute_summary as a (a.key)}
          <div class="flex flex-wrap items-center gap-1 text-xs">
            <span class="text-zinc-400">{a.key} ({a.answered} answered):</span>
            {#each a.values as v (v.value)}
              <span
                class="rounded border border-dashed border-sky-500/50 px-1 text-sky-200"
                >{v.value} {v.count}</span
              >
            {/each}
          </div>
        {/each}
      </div>
    {/if}

    {#if ctl.applied}
      <p
        class="rounded border border-emerald-500/40 bg-emerald-500/10 px-2 py-1 text-xs text-emerald-200"
        data-testid="proposal-applied"
      >
        Labeled {ctl.applied.updated ?? 0} crops as {ctl.applied.class_name}; {ctl.applied
          .skipped_locked} left alone (locked or validated).
      </p>
    {:else}
      {#if ctl.error}
        <p
          class="rounded border border-red-500/40 bg-red-500/10 px-2 py-1 text-xs text-red-200"
          data-testid="proposal-error"
        >
          {ctl.error}
        </p>
        <div class="flex flex-wrap gap-2">
          {#if ctl.errorCode === 'scope_changed'}
            <button
              type="button"
              class="btn"
              data-testid="proposal-anyway"
              onclick={() => void ctl.preview({ allowChangedScope: true })}
            >
              Count the current members anyway
            </button>
          {/if}
          {#if ctl.errorCode === 'class_not_in_registry' || !proposal.in_registry}
            <button
              type="button"
              class="btn"
              data-testid="proposal-add-class"
              onclick={() => (addClassOpen = true)}
            >
              Add class…
            </button>
          {/if}
        </div>
      {/if}
      {#if ctl.dry && chosen}
        <div class="flex flex-wrap items-center gap-3">
          <button
            type="button"
            class="btn btn-primary"
            data-testid="proposal-apply"
            disabled={ctl.busy || ctl.dry.selected === 0}
            onclick={() => void ctl.apply()}
          >
            Apply {chosen} to {ctl.dry.selected} members
          </button>
          <span class="text-xs text-zinc-400" data-testid="proposal-locked">
            {ctl.dry.skipped_locked} left alone: locked, validated or held out. Labels are written
            as a human's.
          </span>
        </div>
      {/if}
    {/if}
  </section>

  <AddClassModal
    open={addClassOpen}
    onclose={() => (addClassOpen = false)}
    oncreated={() => void ctl.preview()}
  />
{/if}
