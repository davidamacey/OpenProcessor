<script lang="ts">
  import Icon from '$components/ui/Icon.svelte';
  /**
   * The Workspace's filter row: the shared item-filter bar, the chips the
   * queue and the scope add to it, and the served next step. Under `md` it
   * sits behind one "Filters" button.
   */
  import { resolve } from '$app/paths';
  import HiddenItemsNote from '$components/itemFilter/HiddenItemsNote.svelte';
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
  class="flex flex-wrap items-center gap-2 border-b border-line bg-surface-1/40 px-4 py-2"
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
  {#if ws.matching}
    <span class="text-xs text-fg-2" data-testid="matching-note">
      <strong class="text-fg-1">{ws.matching.total.toLocaleString()}</strong>
      matching<HiddenItemsNote hidden={ws.matching.hidden} />
    </span>
  {/if}
  {#if scope}
    <button
      type="button"
      class="chip border-accent/60 bg-accent/15 text-accent-text"
      data-testid="scope-chip"
      title="Leave this {scope.kind}"
      onclick={() => ws.setScope(null)}
    >
      {scope.kind === 'cluster' ? 'Cluster' : scope.kind} · {scope.id}<Icon
        name="x"
        size={12}
        class="ml-1 inline-block"
      />
    </button>
  {/if}
  {#if queue}
    <button
      type="button"
      class="chip border-accent/60 bg-accent/15 text-accent-text"
      data-testid="queue-chip"
      title="Leave the {queueLabel} queue"
      onclick={() => ws.setQueue(null)}
    >
      Queue · {queueLabel}<Icon name="x" size={12} class="ml-1 inline-block" />
    </button>
  {/if}
  {#if next && nextHref}
    <a
      class="chip ml-auto border-warn/50 bg-warn/10 text-warn-text"
      href={resolve(nextHref)}
      title={next.detail}
      data-testid="next-step-chip"
    >
      Next: {next.label}
    </a>
  {/if}
</div>
