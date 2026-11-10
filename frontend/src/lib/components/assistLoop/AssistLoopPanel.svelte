<script lang="ts">
  /**
   * The auto-label loop on the clusters page: a collapsible panel to size and
   * start a loop (dry run first), follow it live, and review, audit and roll
   * back what it wrote. Absent when the backend does not serve the loop routes.
   */
  import { onMount } from 'svelte';
  import AssistLoopNodes from '$components/assistLoop/AssistLoopNodes.svelte';
  import AssistLoopRollback from '$components/assistLoop/AssistLoopRollback.svelte';
  import AssistLoopRun from '$components/assistLoop/AssistLoopRun.svelte';
  import AssistLoopStartForm from '$components/assistLoop/AssistLoopStartForm.svelte';
  import {
    AssistLoopController,
    isLiveLoop,
  } from '$lib/assistLoop/assistLoopController.svelte';

  interface Props {
    ctl?: AssistLoopController;
  }

  let { ctl = new AssistLoopController() }: Props = $props();

  let open = $state(false);
  let showForm = $state(false);

  onMount(() => {
    void ctl.load().then(() => {
      // A live loop is what the person came to watch.
      if (ctl.loop && isLiveLoop(ctl.loop.status)) open = true;
    });
    return () => ctl.dispose();
  });

  const hasLoop = $derived(ctl.loop !== null);
  const startable = $derived(!ctl.live);
</script>

{#if ctl.supported}
  <section
    class="border-b border-zinc-800 bg-zinc-900/30 px-4 py-2"
    aria-labelledby="loop-panel-heading"
    data-testid="loop-panel"
  >
    <h2 id="loop-panel-heading" class="text-sm font-medium text-zinc-200">
      <button
        type="button"
        class="flex items-center gap-2 text-left"
        aria-expanded={open}
        aria-controls="loop-panel-body"
        onclick={() => (open = !open)}
        data-testid="loop-panel-toggle"
      >
        Auto-label loop
        {#if ctl.loop}
          <span class="chip font-normal text-zinc-400" data-testid="loop-panel-chip"
            >{ctl.loop.status}</span
          >
        {/if}
      </button>
    </h2>
    {#if open}
      <div id="loop-panel-body" class="mt-3 flex flex-col gap-5 pb-3">
        {#if ctl.loadError}
          <p class="text-xs text-red-300" role="alert" data-testid="loop-load-error">
            {ctl.loadError}
          </p>
        {/if}
        {#if hasLoop}
          <AssistLoopRun {ctl} />
          <AssistLoopNodes {ctl} />
          <AssistLoopRollback {ctl} />
        {/if}
        {#if startable}
          {#if hasLoop && !showForm}
            <div>
              <button
                type="button"
                class="btn btn-sm"
                onclick={() => (showForm = true)}
                data-testid="loop-new"
              >
                Plan a new loop
              </button>
            </div>
          {:else}
            <AssistLoopStartForm {ctl} />
          {/if}
        {/if}
      </div>
    {/if}
  </section>
{/if}
