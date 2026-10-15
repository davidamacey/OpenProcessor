<script lang="ts">
  /**
   * The newest VLM proposal for the viewed cluster or sub-cluster, folded to
   * one summary line so it does not push the grid down; opening it shows the
   * proposal card (which asks the server for the apply dry run only then).
   */
  import AssistProposalCard from '$components/assistRun/AssistProposalCard.svelte';
  import { scopeLabel } from '$lib/assistRun/scopeLabel';
  import type { AssistRunState } from '$lib/types_assist';

  interface Props {
    run: AssistRunState;
    onapplied?: () => void;
  }

  let { run, onapplied }: Props = $props();

  let open = $state(false);
  const proposal = $derived(run.result?.proposal ?? null);
</script>

{#if proposal}
  <details
    class="rounded-md border border-sky-500/30 bg-sky-500/5"
    data-testid="cluster-assist-proposal"
    ontoggle={(e) => (open = e.currentTarget.open)}
  >
    <summary
      class="flex cursor-pointer flex-wrap items-baseline gap-x-2 px-3 py-2 text-sm"
    >
      <span class="font-medium text-zinc-100"
        >VLM proposal for {scopeLabel(run.request.target)}:</span
      >
      <span class="text-zinc-50" data-testid="proposal-summary-class"
        >{proposal.class_name ?? 'no class reached the gate'}</span
      >
      <span class="text-xs text-zinc-400"
        >{Math.round(proposal.agreement * 100)}% of {proposal.answered} answers</span
      >
      {#if run.scope_now?.changed}
        <span class="text-xs text-amber-300">scope changed since</span>
      {/if}
    </summary>
    {#if open}
      <div class="px-3 pb-3">
        <AssistProposalCard {run} {onapplied} />
      </div>
    {/if}
  </details>
{/if}
