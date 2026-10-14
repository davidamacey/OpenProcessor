<!--
  The /review empty-queue state: why the queue is empty, from the served tab
  description, empty reason and sort-fallback reason, with the served next
  step when there is one.
-->
<script lang="ts">
  import { resolve } from '$app/paths';
  import EmptyQueueEmbed from '$components/embedding/EmptyQueueEmbed.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { EmptyQueueMessage } from '$lib/review/reviewCopy';
  import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';

  interface Props {
    emptyMessage: EmptyQueueMessage;
    emptyReason: string | null;
    sortFallbackReason: string | null;
  }

  let { emptyMessage, emptyReason, sortFallbackReason }: Props = $props();
</script>

<!-- R3 (visual audit 2026-09-24): say WHY the queue is empty, from
     the served tab description and sort-fallback reason, instead of
     a bare "Queue empty.". -->
<div class="col-span-full max-w-2xl text-sm" data-testid="queue-empty">
  <p class="text-zinc-300">{emptyMessage.title}</p>
  {#each emptyMessage.lines as line (line)}
    <p class="mt-1 text-xs text-zinc-500">{line}</p>
  {/each}
  {#if emptyMessage.link}
    <!-- #36 item 9: served empty_state says the prerequisite this
         queue needs has never been computed — point straight at the
         control, not just "empty". -->
    <p class="mt-2 text-xs">
      <a
        class="text-blue-400 underline hover:text-blue-300"
        href={resolve(projectHref(emptyMessage.link.href))}>{emptyMessage.link.text}</a
      >
    </p>
  {/if}
  <!-- The served empty_state offers the exact request that embeds the
       items this queue skips; renders only when it applies. -->
  <p class="mt-2 text-xs">
    <EmptyQueueEmbed
      emptyState={reviewTabsVocabularyStore.emptyState}
      reasonText={`${emptyReason ?? ''} ${sortFallbackReason ?? ''}`}
    />
  </p>
</div>
