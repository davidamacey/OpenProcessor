<script lang="ts">
  /**
   * The serving cluster partition, a "Cluster items" run and the activation of a
   * ready partition. Every figure, the decision to serve a run and the refusal of
   * a stale activation are the server's; this prints them.
   */
  import { onMount } from 'svelte';
  import { ClusterRunController } from '$lib/clusters/clusterRunController.svelte';
  import { formatCount } from '$lib/format/count';
  import { formatTimestamp } from '$lib/format/date';
  import type { ClusterTarget } from '$lib/types_clusterList';

  interface Props {
    target?: ClusterTarget;
    ctl?: ClusterRunController;
    /** Called when a different partition starts serving, so a page can reload what it shows. */
    onserving?: () => void;
    /** False when the served next step belongs to another action (one primary per view). */
    primary?: boolean;
  }

  let {
    target = 'item',
    ctl = new ClusterRunController(target),
    onserving,
    primary = true,
  }: Props = $props();

  let lastServing: string | null | undefined;
  $effect(() => {
    const id = ctl.serving?.id ?? null;
    if (lastServing !== undefined && id !== lastServing && id !== null) onserving?.();
    lastServing = id;
  });

  onMount(() => void ctl.load());
</script>

<section
  class="rounded-md border border-line bg-surface-0 p-4"
  data-testid="cluster-run-panel"
>
  <div class="flex flex-wrap items-center justify-between gap-3">
    <div class="min-w-0">
      <h2 class="text-title font-semibold">Clusters</h2>
      <p class="mt-0.5 text-meta text-fg-2" data-testid="cluster-run-serving">
        {#if ctl.loadError}
          <span class="text-danger-text" role="alert">{ctl.loadError}</span>
        {:else if ctl.serving}
          Serving {formatCount(ctl.serving.n_clusters)} clusters over
          {formatCount(ctl.serving.n_items)} items ({ctl.serving.method}, fitted
          {formatTimestamp(ctl.serving.created_at)}).
        {:else}
          No partition is serving yet. Cluster the items to group them.
        {/if}
      </p>
    </div>
    <button
      type="button"
      class="btn {primary ? 'btn-primary' : ''}"
      disabled={ctl.busy}
      onclick={() => void ctl.cluster()}
      data-testid="cluster-run-start"
    >
      {#if ctl.busy && ctl.jobStatus}Clustering ({ctl.jobStatus.replaceAll(
          '_',
          ' ',
        )})…{:else}Cluster items{/if}
    </button>
  </div>

  {#if ctl.pending}
    <div
      class="mt-3 flex flex-wrap items-center gap-3 rounded border border-warn/40 bg-warn/10 p-2 text-meta"
      data-testid="cluster-run-pending"
    >
      <span class="text-warn-text">
        A new partition is ready ({formatCount(ctl.pending.n_clusters)} clusters,
        {formatCount(ctl.pending.n_items)} items) but is not serving.
      </span>
      <button
        type="button"
        class="btn btn-sm"
        disabled={ctl.busy}
        onclick={() => ctl.pending && void ctl.activate(ctl.pending)}
        data-testid="cluster-run-activate"
      >
        Activate it
      </button>
    </div>
  {/if}
  {#if ctl.message}
    <p class="mt-2 text-meta text-fg-2" role="status" data-testid="cluster-run-message">
      {ctl.message}
    </p>
  {/if}
  {#if ctl.actionError}
    <p
      class="mt-2 text-meta text-danger-text"
      role="alert"
      data-testid="cluster-run-error"
    >
      {ctl.actionError}
    </p>
  {/if}
</section>
