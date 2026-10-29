<script lang="ts">
  /**
   * Rollback of what the loop propagated: the served dry run first (what would
   * be restored, what someone changed since and stays), then a confirmation.
   * The server decides what is restorable; this shows its counts.
   */
  import type { AssistLoopController } from '$lib/assistLoop/assistLoopController.svelte';
  import { formatCount } from '$lib/format/count';

  interface Props {
    ctl: AssistLoopController;
  }

  let { ctl }: Props = $props();

  const scopeText = $derived(
    ctl.rollbackNodes === null
      ? 'every cluster the loop labeled'
      : `${ctl.rollbackNodes.join(', ')}`,
  );
</script>

<section
  class="flex flex-col gap-2 text-xs"
  aria-label="Rollback"
  data-testid="loop-rollback"
>
  <div class="flex flex-wrap items-center gap-3">
    <h3 class="text-sm font-medium text-zinc-200">Rollback</h3>
    {#if !ctl.rollbackPlan}
      <button
        type="button"
        class="btn btn-sm"
        disabled={ctl.busy || ctl.live}
        onclick={() => void ctl.previewRollback(null)}
        data-testid="loop-rollback-all"
      >
        Roll back all propagated labels
      </button>
      {#if ctl.live}
        <span class="text-zinc-500">Available once the loop stops.</span>
      {/if}
    {/if}
  </div>

  {#if ctl.rollbackPlan}
    <div
      class="rounded border border-amber-500/40 bg-amber-500/10 p-3"
      role="group"
      aria-label="Confirm rollback"
      data-testid="loop-rollback-plan"
    >
      <p class="text-zinc-100">
        Restore {formatCount(ctl.rollbackPlan.restored)} of
        {formatCount(ctl.rollbackPlan.matched)} items to the class they had before, in
        {scopeText}.
      </p>
      <p class="mt-1 text-zinc-400">
        {formatCount(ctl.rollbackPlan.kept_changed)} were changed since the loop wrote them
        and stay as they are ({formatCount(ctl.rollbackPlan.conflicts)} conflicts). Answers
        the VLM gave directly are kept.
      </p>
      <div class="mt-2 flex gap-2">
        <button
          type="button"
          class="btn btn-sm btn-primary"
          disabled={ctl.busy || ctl.rollbackPlan.restored === 0}
          onclick={() => void ctl.confirmRollback()}
          data-testid="loop-rollback-confirm"
        >
          Roll back {formatCount(ctl.rollbackPlan.restored)} items
        </button>
        <button
          type="button"
          class="btn btn-sm"
          onclick={() => ctl.dismissRollback()}
          data-testid="loop-rollback-dismiss"
        >
          Keep the labels
        </button>
      </div>
    </div>
  {/if}

  {#if ctl.rollbackResult}
    <p class="text-zinc-300" role="status" data-testid="loop-rollback-result">
      Rolled back: {formatCount(ctl.rollbackResult.restored)} restored,
      {formatCount(ctl.rollbackResult.kept_changed)} kept because they changed since,
      {formatCount(ctl.rollbackResult.conflicts)} conflicts.
    </p>
  {/if}
</section>
