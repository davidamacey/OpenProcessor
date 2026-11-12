<script lang="ts">
  /**
   * One whole image: the served thumbnail with the served boxes over it, the served counts as
   * chips, and the near-duplicate badge when the server says the image is in a group. Nothing is
   * counted or labelled here.
   */
  import { resolveApiUrl } from '$lib/api';
  import type { ImageCardWire } from '$lib/types_images';
  import BoxLayer from '$components/boxes/BoxLayer.svelte';

  /** Below this card width the boxes draw without their labels. */
  const LABEL_MIN_WIDTH = 96;

  interface Props {
    card: ImageCardWire;
    selected: boolean;
    focused: boolean;
    onclick: (e: MouseEvent) => void;
    onopen: () => void;
  }

  let { card, selected, focused, onclick, onopen }: Props = $props();

  let width = $state(0);
  const aspect = $derived(
    card.width && card.height ? `${card.width} / ${card.height}` : '4 / 3',
  );
</script>

<li
  class="flex flex-col rounded-md border bg-zinc-900 {selected
    ? 'border-sky-400 bg-zinc-800'
    : 'border-zinc-700'} {focused ? 'ring-2 ring-sky-500 ring-inset' : ''}"
  style="content-visibility:auto;contain-intrinsic-size:auto 220px"
  data-testid="image-card"
  data-image-id={card.image_id}
  data-selected={selected}
  aria-current={focused ? 'true' : undefined}
>
  <div class="relative" bind:clientWidth={width}>
    <button
      type="button"
      class="relative block w-full overflow-hidden rounded-t bg-zinc-950"
      style="aspect-ratio:{aspect}"
      data-testid="image-card-select"
      aria-pressed={selected}
      aria-label="Select {card.file_name}"
      {onclick}
      ondblclick={onopen}
    >
      <img
        src={resolveApiUrl(card.thumbnail_url)}
        alt=""
        loading="lazy"
        class="h-full w-full object-fill"
      />
      <BoxLayer boxes={card.boxes} showLabels={width >= LABEL_MIN_WIDTH} />
    </button>
    {#if selected}
      <span
        class="pointer-events-none absolute top-1 left-1 flex h-5 w-5 items-center justify-center rounded-full bg-sky-500 text-xs text-white"
        aria-hidden="true">✓</span
      >
    {/if}
    {#if card.near_dup}
      <span
        class="chip absolute top-1 right-1"
        data-testid="near-dup-badge"
        title="Near-duplicate group ({card.near_dup.purpose})"
      >
        Near-dup{card.near_dup.size != null ? ` · ${card.near_dup.size}` : ''}
      </span>
    {/if}
  </div>
  <div class="flex flex-wrap items-center gap-1 px-2 py-1.5 text-xs">
    <span class="chip" data-testid="chip-items" title="Items on this image"
      >{card.counts.items} items</span
    >
    <span class="chip" data-testid="chip-labeled" title="Items with a class"
      >{card.counts.labeled} labelled</span
    >
    <span
      class="chip {card.counts.unlabeled > 0 ? 'unlabeled-chip' : ''}"
      data-testid="chip-unlabeled"
      title="Items without a class">{card.counts.unlabeled} unlabelled</span
    >
    {#if card.boxes_truncated}
      <span
        class="chip"
        data-testid="boxes-truncated"
        title="This image has more items than are drawn">+ more boxes</span
      >
    {/if}
  </div>
  <div
    class="flex items-center justify-between gap-2 px-2 pb-1.5 text-[11px] text-zinc-400"
  >
    <span class="truncate" title={card.file_name}>{card.file_name}</span>
    <button
      type="button"
      class="shrink-0 underline hover:text-zinc-200"
      data-testid="image-card-open"
      onclick={onopen}>Details</button
    >
  </div>
</li>

<style>
  .unlabeled-chip {
    border-color: rgb(var(--system-orange));
    color: rgb(var(--system-orange));
  }
</style>
