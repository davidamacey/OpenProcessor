<script lang="ts">
  /**
   * Label operations: the bulk labels and moves the server recorded for this
   * project, newest first, and undo. An undo shows the server's dry run
   * (what it would restore, what it would keep, the conflicts) before it
   * offers Confirm. Every figure is served.
   */
  import { onMount } from 'svelte';
  import LabelOpDialog from '$lib/components/labelOps/LabelOpDialog.svelte';
  import LabelOpsList from '$lib/components/labelOps/LabelOpsList.svelte';
  import { LabelOpsController } from '$lib/labelOps/labelOpsController.svelte';

  const ctl = new LabelOpsController();

  onMount(() => {
    const ac = new AbortController();
    void ctl.load(ac.signal);
    return () => ac.abort();
  });
</script>

<div class="mx-auto flex h-full max-w-5xl flex-col gap-4 p-4 sm:p-6">
  <header class="flex flex-wrap items-center gap-3">
    <h1 class="text-2xl font-semibold tracking-tight">Label operations</h1>
    <span class="grow"></span>
    <button
      type="button"
      class="btn"
      disabled={ctl.loading}
      onclick={() => void ctl.load()}>Refresh</button
    >
  </header>
  <p class="text-xs text-zinc-400">
    Bulk labels and moves recorded for this project. An undo restores only the items still
    carrying the operation's label; anything edited since, in the test holdout or whose
    restore class was renamed is kept as it is.
    {#if ctl.retention != null}The newest {ctl.retention} are kept.{/if}
  </p>

  {#if ctl.loading && ctl.items.length === 0 && !ctl.loadError}
    <section class="surface p-6 text-sm text-zinc-500">Loading…</section>
  {:else if ctl.loadError}
    <section class="surface flex flex-col gap-3 p-6 text-sm">
      <p class="text-red-300" role="alert" data-testid="label-ops-error">
        {ctl.loadError}
      </p>
      <button type="button" class="btn w-fit" onclick={() => void ctl.load()}
        >Retry</button
      >
    </section>
  {:else if ctl.items.length === 0}
    <section class="surface p-6 text-sm text-zinc-400" data-testid="label-ops-empty">
      No label operations recorded yet. A bulk label or move from a cluster shows up here.
    </section>
  {:else}
    <LabelOpsList items={ctl.items} onopen={(id) => ctl.open(id)} />
    {#if ctl.canShowMore}
      <button
        type="button"
        class="btn w-fit self-center"
        disabled={ctl.loading}
        data-testid="label-ops-more"
        onclick={() => void ctl.showMore()}>Show more</button
      >
    {/if}
  {/if}

  {#if ctl.selected}
    <LabelOpDialog {ctl} op={ctl.selected} />
  {/if}
</div>
