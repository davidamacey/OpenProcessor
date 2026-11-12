<script lang="ts">
  /** The gallery grid: the cards in the order given, with the next page loaded on scroll. */
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import type { ImageCardWire } from '$lib/types_images';
  import ImageCard from './ImageCard.svelte';

  type Thumb = 's' | 'm' | 'l';
  const MIN_COLUMN: Record<Thumb, string> = { s: '120px', m: '180px', l: '260px' };

  interface Props {
    cards: ImageCardWire[];
    thumb?: Thumb;
    isSelected: (imageId: string) => boolean;
    focusedId?: string | null;
    hasMore: boolean;
    loadingMore: boolean;
    onselect: (imageId: string, e: MouseEvent) => void;
    onopen: (imageId: string) => void;
    onloadmore: () => void;
  }

  let {
    cards,
    thumb = 'm',
    isSelected,
    focusedId = null,
    hasMore,
    loadingMore,
    onselect,
    onopen,
    onloadmore,
  }: Props = $props();
</script>

<ul
  class="grid gap-3"
  style="grid-template-columns:repeat(auto-fill,minmax({MIN_COLUMN[
    thumb
  ]},1fr));grid-auto-rows:1fr"
  data-testid="image-grid"
>
  {#each cards as card (card.image_id)}
    <ImageCard
      {card}
      selected={isSelected(card.image_id)}
      focused={focusedId === card.image_id}
      onclick={(e) => onselect(card.image_id, e)}
      onopen={() => onopen(card.image_id)}
    />
  {/each}
</ul>
<div
  use:infiniteScroll={{ onload: onloadmore, disabled: !hasMore || loadingMore }}
  class="h-1"
  aria-hidden="true"
></div>
