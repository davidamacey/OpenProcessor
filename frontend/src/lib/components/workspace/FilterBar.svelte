<script lang="ts">
  /**
   * The Workspace's filter row: the shared item-filter bar, the chips the
   * queue and the scope add to it, and the served next step. Under `md` it
   * sits behind one "Filters" button.
   */
  import { resolve } from '$app/paths';
  import ItemFilterBar from '$components/itemFilter/ItemFilterBar.svelte';
  import { withoutOpenVocab } from '$lib/itemFilter/itemFilterState.svelte';
  import { projectHref, type ProjectSubPath } from '$lib/projectPaths';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import { nextStepsStore } from '$stores/nextSteps.svelte';
  import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';

  let { ws }: { ws: WorkspaceController } = $props();

  let open = $state(false);

  const queue = $derived(ws.url.queue);
  const scope = $derived(ws.url.scope);
  const queueLabel = $derived(
    reviewTabsVocabularyStore.list.find((t) => t.id === queue)?.label ?? queue,
  );
  const next = $derived(nextStepsStore.next);
  const nextHref = $derived(
    next
      ? projectHref(
          `/${next.section}${next.query ? `?${next.query}` : ''}` as ProjectSubPath,
        )
      : null,
  );
  const showOpenVocab = $derived(ws.matchingMode);
  const visible = (p: string): boolean =>
    p !== 'max_rank' && (showOpenVocab || withoutOpenVocab(p));
</script>

<div
  class="flex flex-wrap items-center gap-2 border-b border-zinc-800 bg-zinc-900/40 px-4 py-2"
  data-testid="workspace-filter-bar"
>
  <button
    type="button"
    class="btn md:hidden"
    aria-expanded={open}
    aria-controls="workspace-filters"
    data-testid="filters-toggle"
    onclick={() => (open = !open)}
  >
    Filters
  </button>
  <div id="workspace-filters" class="contents max-md:[&:not(.open)]:hidden" class:open>
    <ItemFilterBar
      state={ws.filter}
      {visible}
      {showOpenVocab}
      onchange={() => ws.applyFilter()}
      rootClass="contents"
    />
  </div>
  {#if scope}
    <button
      type="button"
      class="chip border-blue-500/60 bg-blue-500/15 text-blue-100"
      data-testid="scope-chip"
      title="Leave this {scope.kind}"
      onclick={() => ws.setScope(null)}
    >
      {scope.kind === 'cluster' ? 'Cluster' : scope.kind} · {scope.id} ×
    </button>
  {/if}
  {#if queue}
    <button
      type="button"
      class="chip border-blue-500/60 bg-blue-500/15 text-blue-100"
      data-testid="queue-chip"
      title="Leave the {queueLabel} queue"
      onclick={() => ws.setQueue(null)}
    >
      Queue · {queueLabel} ×
    </button>
  {/if}
  {#if next && nextHref}
    <a
      class="chip ml-auto border-amber-500/50 bg-amber-500/10 text-amber-100"
      href={resolve(nextHref)}
      title={next.detail}
      data-testid="next-step-chip"
    >
      Next: {next.label}
    </a>
  {/if}
</div>
