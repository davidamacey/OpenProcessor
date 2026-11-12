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
  <h2 class="text-xs font-semibold tracking-wide text-zinc-400 uppercase">Cluster</h2>
  {#if card}
    <p class="text-zinc-100">
      Cluster {displayNoText(card)}
      <span class="font-mono text-xs text-zinc-500">{card.uid}</span>
    </p>
    <p class="text-zinc-300">
      size {card.size.toLocaleString()} · dominant
      <strong class="text-zinc-100">{card.dominant_class_name ?? '—'}</strong>
    </p>
    {#if card.label_agreement != null}
      <p class="font-mono text-xs text-zinc-400" data-testid="cluster-agreement">
        agreement {(card.label_agreement * 100).toFixed(0)}%
        {#if card.labelled_share != null}
          · labelled {(card.labelled_share * 100).toFixed(0)}%
        {/if}
      </p>
    {/if}
    <p
      class="font-mono text-[11px] text-zinc-300"
      data-testid="cluster-header-counts"
      title="Class-wide counts are every crop of this class (test holdout included), from the class registry; 'in this cluster' is this cluster's own member count."
    >
      {#if cls}
        <span class="text-zinc-500">class-wide:</span>
        <span class="text-emerald-300">{(cls.validated_count ?? 0).toLocaleString()}</span
        >
        <span class="text-zinc-500">validated</span>
        <span class="text-blue-300">{(cls.count ?? 0).toLocaleString()}</span>
        <span class="text-zinc-500">labeled</span> |
      {/if}
      <span class="text-zinc-200">{card.size.toLocaleString()}</span>
      <span class="text-zinc-500">in this cluster</span>
    </p>
  {:else}
    <p class="text-zinc-500">Open a cluster to see its details.</p>
  {/if}
</section>
