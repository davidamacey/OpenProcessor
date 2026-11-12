<script lang="ts">
  /**
   * The rail: the served review queues with their counts, then the classes
   * with theirs. Both lists are rendered as served, in the served order. A
   * drawer under `lg`.
   */
  import RailClasses from './RailClasses.svelte';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';

  let { ws }: { ws: WorkspaceController } = $props();

  $effect(() => {
    void reviewTabsVocabularyStore.init();
  });

  const queues = $derived(reviewTabsVocabularyStore.list.filter((t) => t.available));
</script>

<aside
  class="{ws.railOpen
    ? 'fixed inset-y-0 left-0 z-30 w-72 max-w-full'
    : 'hidden'} flex-col border-r border-zinc-800 bg-zinc-950 lg:static lg:z-auto lg:flex lg:w-64 lg:shrink-0"
  aria-label="Queues and classes"
  data-testid="workspace-rail"
>
  <div class="flex justify-end p-2 lg:hidden">
    <button type="button" class="btn" onclick={() => ws.toggleRail()}>Close</button>
  </div>
  <section aria-label="Queues" class="border-b border-zinc-800 pb-2">
    <h2 class="p-3 text-xs font-semibold tracking-wide text-zinc-400 uppercase">
      Queues
    </h2>
    <ul>
      {#each queues as q (q.id)}
        <li>
          <button
            type="button"
            class="flex min-h-9 w-full items-center justify-between gap-2 px-3 py-1.5 text-left text-sm hover:bg-zinc-900 {ws
              .url.queue === q.id
              ? 'bg-zinc-800 text-white'
              : 'text-zinc-300'}"
            aria-current={ws.url.queue === q.id ? 'true' : undefined}
            title={q.description}
            data-testid="queue-{q.id}"
            onclick={() => ws.setQueue(ws.url.queue === q.id ? null : q.id)}
          >
            <span class="truncate">{q.label}</span>
            <span class="font-mono text-xs text-zinc-400">{q.count.toLocaleString()}</span
            >
          </button>
        </li>
      {/each}
    </ul>
  </section>
  <RailClasses {ws} />
</aside>
