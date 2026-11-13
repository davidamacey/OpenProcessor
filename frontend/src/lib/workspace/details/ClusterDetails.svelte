<script lang="ts">
  /** The open cluster's served card: size, dominant class, label agreement and counts. */
  import Button from '$components/ui/Button.svelte';
  import { displayNameNote, displayNoText } from '$lib/clusters/clusterCardText';
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

<section class="space-y-2 text-body" data-testid="cluster-details">
  <h2 class="text-meta font-semibold text-fg-2">Cluster</h2>
  {#if card}
    <p class="text-fg-1">
      Cluster {displayNoText(card)}
      <code class="text-meta text-fg-3">{card.uid}</code>
    </p>
    <p class="text-fg-2" data-testid="cluster-details-name">
      size {card.size.toLocaleString()} ·
      {#if card.display_name_kind === 'none'}
        <strong class="text-fg-1">not named yet</strong>
      {:else}
        <strong class="text-fg-1">{card.display_name}</strong>
        {#if displayNameNote(card.display_name_kind)}
          <span class="text-fg-3">({displayNameNote(card.display_name_kind)})</span>
        {/if}
      {/if}
    </p>
    <!-- The main path: name the whole group in one action. The class picker belongs to the bulk
         bar, the selection is "everything this cluster scope matches", and the server counts it. -->
    <Button
      icon="pencil"
      data-testid="label-cluster"
      onclick={() => {
        ws.confirmSelectAllMatching();
        ws.pickerOpen = true;
      }}>Label all {card.size.toLocaleString()} in this cluster…</Button
    >
    {#if card.label_agreement != null}
      <p class="text-meta text-fg-2 tabular-nums" data-testid="cluster-agreement">
        agreement {(card.label_agreement * 100).toFixed(0)}%
        {#if card.labelled_share != null}
          · labelled {(card.labelled_share * 100).toFixed(0)}%
        {/if}
      </p>
    {/if}
    <p
      class="text-meta text-fg-2 tabular-nums"
      data-testid="cluster-header-counts"
      title="validated · labeled · cluster total — class-wide counts are every crop of this class (test holdout included), from the class registry; 'in this cluster' is this cluster's own member count (test holdout included). The grid lists only reviewable members."
    >
      {#if cls}
        <span class="text-fg-3">class-wide:</span>
        <span class="text-success-text"
          >{(cls.validated_count ?? 0).toLocaleString()}</span
        >
        <span class="text-fg-3">validated</span>
        <span class="mx-1 text-fg-3">·</span>
        <span class="text-accent-text">{(cls.count ?? 0).toLocaleString()}</span>
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
