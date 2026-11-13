<script lang="ts">
  import Icon from '$components/ui/Icon.svelte';
  /**
   * The Workspace's filter row: the shared item-filter bar, the chips the
   * queue and the scope add to it, and the served next step. Under `md` it
   * sits behind one "Filters" button.
   */
  import { resolve } from '$app/paths';
  import { getImage } from '$lib/api_images';
  import HiddenItemsNote from '$components/itemFilter/HiddenItemsNote.svelte';
  import ItemFilterBar from '$components/itemFilter/ItemFilterBar.svelte';
  import { withoutOpenVocab } from '$lib/itemFilter/itemFilterState.svelte';
  import { projectHref, type ProjectSubPath } from '$lib/projectPaths';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import { nextStepsStore } from '$stores/nextSteps.svelte';
  import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';

  let { ws }: { ws: WorkspaceController } = $props();

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
  // The scope reads as its served name (the image's file name, the cluster's display name); the
  // id is in the chip's tooltip.
  let imageName = $state<{ id: string; name: string } | null>(null);
  $effect(() => {
    const s = scope;
    if (s?.kind !== 'image' || imageName?.id === s.id) return;
    const ctl = new AbortController();
    getImage(s.id, ctl.signal)
      .then((img) => (imageName = { id: s.id, name: img.image.file_name }))
      .catch(() => {});
    return () => ctl.abort();
  });
  const scopeLabel = $derived.by(() => {
    if (!scope) return '';
    if (scope.kind === 'cluster')
      return `Cluster: ${ws.scopeCard?.display_name || scope.id}`;
    if (scope.kind === 'image')
      return `Image: ${imageName?.id === scope.id ? imageName.name : scope.id}`;
    return `${scope.kind}: ${scope.id}`;
  });
  // Under `md` the row is there only when it has something to say: the controls opened from the
  // view bar, a chip, a scope, a queue or the next step.
  const quiet = $derived(
    !ws.filtersOpen && !scope && !queue && !next && ws.filter.chips().length === 0,
  );
  const showOpenVocab = $derived(ws.matchingMode);
  const visible = (p: string): boolean =>
    p !== 'max_rank' && (showOpenVocab || withoutOpenVocab(p));
</script>

<div
  class="flex flex-wrap items-center gap-2 border-b border-line bg-surface-1/40 px-4 py-2 {quiet
    ? 'max-md:hidden'
    : ''}"
  data-testid="workspace-filter-bar"
>
  <div
    id="workspace-filters"
    class="contents max-md:[&:not(.open)]:hidden"
    class:open={ws.filtersOpen}
  >
    <ItemFilterBar
      state={ws.filter}
      {visible}
      {showOpenVocab}
      onchange={() => ws.applyFilter()}
      rootClass="contents"
    />
  </div>
  {#if ws.matching}
    <span class="text-meta text-fg-2" data-testid="matching-note">
      <strong class="text-fg-1">{ws.matching.total.toLocaleString()}</strong>
      matching<HiddenItemsNote hidden={ws.matching.hidden} />
    </span>
  {/if}
  {#if scope}
    <button
      type="button"
      class="chip border-accent/60 bg-accent/15 text-accent-text"
      data-testid="scope-chip"
      title="Leave this {scope.kind} ({scope.id})"
      onclick={() => ws.setScope(null)}
    >
      {scopeLabel}<Icon name="x" size={12} class="ml-1 inline-block" />
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
      class="chip chip-warn ml-auto"
      href={resolve(nextHref)}
      title={next.detail}
      data-testid="next-step-chip"
    >
      Next: {next.label}
    </a>
  {/if}
</div>
