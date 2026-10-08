<script lang="ts">
  /**
   * "Matching items" on /clusters: every item the shared filter matches, as a
   * flat crop grid, with run-on-selection actions (ignore, restore, label,
   * move) that act on all of them through the server's dry-run-first
   * selection writes. The counts shown are the server's.
   */
  import { untrack } from 'svelte';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import CropResultGrid from '$lib/components/CropResultGrid.svelte';
  import SelectionActionDialog from './SelectionActionDialog.svelte';
  import { createMatchingItems } from '$lib/itemFilter/matchingItems.svelte';
  import { createSelectionActionController } from '$lib/itemFilter/selectionActionController.svelte';
  import { createSelection } from '$lib/selection.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { undoStore } from '$stores/undo.svelte';
  import type { Crop } from '$lib/types';
  import type { ItemFilter, ItemFilterQuery } from '$lib/types_itemFilter';

  interface Props {
    /** The filter as query parameters (for the list) and as a body (for the
     *  selection writes); the same filter, spelled twice by its owner. */
    query: () => ItemFilterQuery;
    body: () => ItemFilter;
    onexit: () => void;
    ondetail?: (crop: Crop) => void;
  }

  let { query, body, onexit, ondetail }: Props = $props();

  const matching = createMatchingItems(() => query());
  const pager = matching.pager;
  const sel = createSelection({ plainClick: 'replace' });
  const actions = createSelectionActionController(() => body());

  const queryKey = $derived(JSON.stringify(query()));
  $effect(() => {
    void queryKey;
    // loadFirst() reads and writes the pager's own state; untracked so this
    // effect depends on the filter alone.
    untrack(() => {
      sel.clear();
      void pager.loadFirst();
    });
  });

  // Z reverts the last write (a selection write records its served ids), then
  // the list reloads: what the write touched may now match or not match.
  $effect(() => {
    const off = keyboardStore.registerAction(
      'clusters_search.undo',
      () => {
        void undoStore.undoLast().then(() => pager.loadFirst());
      },
      'clusters',
    );
    return off;
  });

  const ACTIONS = [
    { id: 'exclude', label: 'Ignore all matching' },
    { id: 'unexclude', label: 'Restore all matching' },
    { id: 'label', label: 'Label all matching…' },
    { id: 'move', label: 'Move all matching…' },
  ] as const;
</script>

<div class="mb-3 flex flex-wrap items-center gap-2 text-xs" data-testid="matching-header">
  <span class="text-zinc-300">
    <strong class="text-zinc-100" data-testid="matching-total"
      >{pager.total.toLocaleString()}</strong
    >
    matching item{pager.total === 1 ? '' : 's'}
  </span>
  <span class="grow"></span>
  {#each ACTIONS as a (a.id)}
    <button
      type="button"
      class="btn-sm border border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
      data-testid="matching-action-{a.id}"
      disabled={pager.loading}
      onclick={() => void actions.open(a.id)}
    >
      {a.label}
    </button>
  {/each}
  <button
    type="button"
    class="btn-sm border border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
    onclick={onexit}
  >
    ← Back to clusters
  </button>
</div>

{#if pager.loading && pager.items.length === 0}
  <p class="text-sm text-zinc-500">Loading...</p>
{:else if pager.error}
  <p class="text-sm text-red-300" role="alert" data-testid="matching-error">
    {pager.error}
  </p>
{:else if pager.items.length === 0}
  <p class="text-sm text-zinc-500">No items match this filter.</p>
{:else}
  <CropResultGrid items={pager.items} {sel} {ondetail} />
  <div
    use:infiniteScroll={{
      onload: () => pager.loadMore(),
      disabled: !pager.hasMore || pager.loading || pager.loadingMore,
    }}
    class="mt-4 h-1"
    aria-hidden="true"
  ></div>
  <p class="mt-2 font-mono text-xs text-zinc-500" data-testid="matching-footer">
    {pager.items.length} / {pager.total}
  </p>
{/if}

<SelectionActionDialog controller={actions} onapplied={() => void pager.loadFirst()} />
