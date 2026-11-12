<script lang="ts">
  /** The open cluster's served card: size, dominant class, label agreement and counts. */
  import { displayNoText } from '$lib/clusters/clusterCardText';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import { classesStore } from '$stores/classes.svelte';

  let { ws }: { ws: WorkspaceController; id: string | null } = $props();

  const card = $derived(ws.scopeCard);
  const cls = $derived(
    card?.dominant_class_name != null
      ? (classesStore.classes.find((c) => c.name === card.dominant_class_name) ?? null)
      : null,
  );
</script>

<section class="space-y-2 text-sm" data-testid="cluster-details">
  <h2 class="text-xs font-semibold tracking-wide text-fg-2 uppercase">Cluster</h2>
  {#if card}
    <p class="text-fg-1">
      Cluster {displayNoText(card)}
      <span class="font-mono text-xs text-fg-3">{card.uid}</span>
    </p>
    <p class="text-fg-2">
      size {card.size.toLocaleString()} · dominant
      <strong class="text-fg-1">{card.dominant_class_name ?? '—'}</strong>
    </p>
    {#if card.label_agreement != null}
      <p class="font-mono text-xs text-fg-2" data-testid="cluster-agreement">
        agreement {(card.label_agreement * 100).toFixed(0)}%
        {#if card.labelled_share != null}
          · labelled {(card.labelled_share * 100).toFixed(0)}%
        {/if}
      </p>
    {/if}
    <p
      class="font-mono text-caption text-fg-2"
      data-testid="cluster-header-counts"
      title="validated · labeled · cluster total — class-wide counts are every crop of this class (test holdout included), from the class registry; 'in this cluster' is this cluster's own member count (test holdout included). The grid lists only reviewable members."
    >
      {#if cls}
        <span class="text-fg-3">class-wide:</span>
        <span class="text-success">{(cls.validated_count ?? 0).toLocaleString()}</span>
        <span class="text-fg-3">validated</span>
        <span class="mx-1 text-fg-3">·</span>
        <span class="text-accent">{(cls.count ?? 0).toLocaleString()}</span>
        <span class="text-fg-3">labeled</span>
        <span class="mx-1 text-fg-3">|</span>
      {/if}
      <span class="text-fg-1">{card.size.toLocaleString()}</span>
      <span class="text-fg-3">in this cluster</span>
    </p>
  {:else}
    <p class="text-fg-3">Open a cluster to see its details.</p>
  {/if}
</section>
