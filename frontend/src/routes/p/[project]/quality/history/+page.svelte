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

<div class="flex flex-col gap-4">
  <header class="flex flex-wrap items-center gap-3">
    <h2 class="text-title font-semibold text-fg-1">Label operations</h2>
    <span class="grow"></span>
    <button
      type="button"
      class="btn"
      disabled={ctl.loading}
      onclick={() => void ctl.load()}>Refresh</button
    >
  </header>
  <p class="text-meta text-fg-2">
    Labels and moves recorded for this project, single or bulk. An undo restores only the
    items still carrying the operation's label; anything edited since, in the test holdout
    or whose restore class was renamed is kept as it is.
    {#if ctl.retention != null && ctl.singleRetention != null}The newest {ctl.retention} bulk
      operations and the newest {ctl.singleRetention} single labels are kept.{/if}
  </p>

  {#if ctl.loading && ctl.items.length === 0 && !ctl.loadError}
    <section class="surface p-6 text-body text-fg-3">Loading…</section>
  {:else if ctl.loadError}
    <section class="surface flex flex-col gap-3 p-6 text-body">
      <p class="text-danger-text" role="alert" data-testid="label-ops-error">
        {ctl.loadError}
      </p>
      <button type="button" class="btn w-fit" onclick={() => void ctl.load()}
        >Retry</button
      >
    </section>
  {:else if ctl.items.length === 0}
    <section class="surface p-6 text-body text-fg-2" data-testid="label-ops-empty">
      No label operations recorded yet. A bulk label or move from a cluster shows up here.
    </section>
  {:else}
    <LabelOpsList items={ctl.items} onopen={(id) => ctl.open(id)} />
    {#if ctl.olderError}
      <p
        role="alert"
        data-testid="label-ops-older-error"
        class="text-body text-danger-text self-center"
      >
        {ctl.olderError}
      </p>
    {/if}
    {#if ctl.canLoadOlder}
      <button
        type="button"
        class="btn w-fit self-center"
        disabled={ctl.loadingOlder}
        data-testid="label-ops-older"
        onclick={() => void ctl.loadOlder()}
        >{ctl.loadingOlder ? 'Loading…' : 'Load older'}</button
      >
    {:else}
      <p data-testid="label-ops-end" class="text-meta text-fg-3 self-center">
        No older operations.
      </p>
    {/if}
  {/if}

  {#if ctl.selected}
    <LabelOpDialog {ctl} op={ctl.selected} />
  {/if}
</div>
