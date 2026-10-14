<!--
  The /review "Details" disclosure: history, source image, siblings and
  item-text lines through CropMetaPanel.
-->
<script lang="ts">
  import CropMetaPanel from '$lib/components/CropMetaPanel.svelte';
  import type { Pager } from '$lib/pager.svelte';
  import type { ReviewItem } from '$lib/types_review';

  interface Props {
    current: ReviewItem;
    queue: Pager<ReviewItem>;
    detailsOpen: boolean;
  }

  let { current, queue, detailsOpen = $bindable() }: Props = $props();
</script>

<!-- G7/G9/G8: history + source image/siblings + item-text lines,
   via the same CropMetaPanel used by the /clusters detail
   modal — collapsed by default so it doesn't compete with the
   confirm/reject flow above. -->
<div class="mt-3 border-t border-zinc-800 pt-2">
  <button
    type="button"
    class="text-[10px] uppercase tracking-wider text-zinc-500 hover:text-zinc-300"
    onclick={() => (detailsOpen = !detailsOpen)}
  >
    {detailsOpen ? '▾' : '▸'} Details
  </button>
  {#if detailsOpen}
    <div class="mt-2">
      <CropMetaPanel
        crop={current}
        embedded
        onreprocessed={(c) => {
          const idx = queue.items.findIndex((x) => x.id === c.id);
          if (idx >= 0) queue.items[idx] = { ...queue.items[idx], ...c } as ReviewItem;
        }}
      />
    </div>
  {/if}
</div>
