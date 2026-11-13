<script lang="ts">
  /**
   * The focused map cell: the server's summary of the photos in it (`POST /selection/summary`
   * with the cell's bounds as `geo_bbox`) and its first image cards (`GET /images`). One request
   * pair per focused cell; the cards link to the image details.
   */
  import ErrorState from '$components/ui/ErrorState.svelte';
  import { apiErrorText } from '$lib/api_errors';
  import { activeMap } from '$lib/map/activeMap.svelte';
  import { CELL_CARDS } from '$lib/map/mapController.svelte';
  import { VIEWS } from '$lib/workspace/viewRegistry';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let { ws }: { ws: WorkspaceController; id: string | null } = $props();

  const ctl = $derived(activeMap.ctl);
  const cell = $derived(ctl?.cell ?? null);
  const target = $derived(
    VIEWS.find((v) => v.id === 'gallery') ?? VIEWS.find((v) => v.id === 'detections'),
  );
</script>

<section class="space-y-2 text-body" data-testid="cell-details">
  <h2 class="text-meta font-semibold text-fg-2">Map cell</h2>
  {#if !ctl || !cell}
    <p class="text-fg-3">Select a cell on the map to see its photos.</p>
  {:else if ctl.cellFailure}
    <ErrorState
      title="This cell could not be read"
      error={ctl.cellFailure}
      message={apiErrorText(ctl.cellFailure)}
      onretry={() => void ctl.retryCell()}
    />
  {:else if !ctl.summary || !ctl.cards}
    <p class="text-fg-3" role="status">Loading…</p>
  {:else}
    <p class="text-fg-1" data-testid="cell-summary">{ctl.summary.summary_text}</p>
    <p class="text-meta text-fg-2">
      {ctl.summary.images.toLocaleString()} photo{ctl.summary.images === 1 ? '' : 's'}
      here
      {#if ctl.cards.total > CELL_CARDS}· showing the first {CELL_CARDS}{/if}
    </p>
    {#if ctl.summary.join_truncated}
      <p class="text-meta text-warn-text">
        The cell holds more photos than can be counted at once; zoom in to narrow it.
      </p>
    {/if}
    <ul class="grid grid-cols-3 gap-1" data-testid="cell-cards">
      {#each ctl.cards.items as card (card.image_id)}
        <li>
          <button
            type="button"
            class="block w-full overflow-hidden rounded border border-line"
            title={card.file_name}
            data-testid="cell-card"
            onclick={() =>
              ws.setFocus({ kind: 'image', id: card.image_id }, { push: true })}
          >
            <img
              src={card.thumbnail_url}
              alt={card.file_name}
              class="aspect-square w-full object-cover"
              loading="lazy"
            />
          </button>
        </li>
      {/each}
    </ul>
    <div class="flex gap-2">
      {#if target}
        <button
          type="button"
          class="btn"
          data-testid="cell-show"
          onclick={() => ctl.showCell(cell, target.id)}>Show in {target.label}</button
        >
      {/if}
      <button type="button" class="btn" onclick={() => ctl.clearCell()}>Close</button>
    </div>
  {/if}
</section>
