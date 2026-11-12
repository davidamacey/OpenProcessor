<script lang="ts">
  import Icon from '$components/ui/Icon.svelte';
  /**
   * One whole image: the served thumbnail with the served boxes over it, the served counts as
   * chips, and the near-duplicate badge when the server says the image is in a group. Nothing is
   * counted or labelled here.
   */
  import { resolveApiUrl } from '$lib/api';
  import { fetchSourceMissing, type SourceMissing } from '$lib/sourceMissing';
  import SourceMissingBadge from '$components/SourceMissingBadge.svelte';
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
  // A thumbnail that fails is asked for once more after a second; a second failure asks the
  // server why and shows its answer.
  const RETRY_AFTER_MS = 1000;
  let attempt = $state(0);
  let missing = $state<SourceMissing | null>(null);

  function thumbFailed(e: Event): void {
    if (attempt === 0) {
      setTimeout(() => (attempt = 1), RETRY_AFTER_MS);
      return;
    }
    void fetchSourceMissing((e.currentTarget as HTMLImageElement).src).then(
      (m) => (missing = m),
    );
  }
  // Every thumbnail sits in the same 4:3 frame, so cards are one height; the picture (and the
  // boxes drawn over it) keeps its own aspect inside the frame.
  const FRAME = 4 / 3;
  const picture = $derived.by(() => {
    const ratio = card.width && card.height ? card.width / card.height : FRAME;
    const fit = ratio >= FRAME ? 'width:100%' : 'height:100%';
    return `aspect-ratio:${ratio};${fit}`;
  });
</script>

<li
  class="flex h-full flex-col rounded-md border bg-surface-1 {selected
    ? 'border-info bg-surface-2'
    : 'border-line-strong'} {focused ? 'ring-2 ring-info ring-inset' : ''}"
  style="content-visibility:auto;contain-intrinsic-size:auto 220px"
  data-testid="image-card"
  data-image-id={card.image_id}
  data-selected={selected}
  aria-current={focused ? 'true' : undefined}
>
  <div class="relative" bind:clientWidth={width}>
    <button
      type="button"
      class="relative flex w-full items-center justify-center overflow-hidden rounded-t bg-surface-0"
      style="aspect-ratio:4 / 3"
      data-testid="image-card-select"
      aria-pressed={selected}
      aria-label="Select {card.file_name}"
      {onclick}
      ondblclick={onopen}
    >
      <span class="relative block" style={picture}>
        {#key attempt}
          <img
            src={resolveApiUrl(card.thumbnail_url)}
            alt=""
            loading="lazy"
            class="h-full w-full object-fill"
            onerror={thumbFailed}
          />
        {/key}
        {#if missing}
          <SourceMissingBadge {missing} compact />
        {/if}
        <BoxLayer boxes={card.boxes} showLabels={width >= LABEL_MIN_WIDTH} />
      </span>
    </button>
    {#if selected}
      <span
        class="pointer-events-none absolute top-1 left-1 flex h-5 w-5 items-center justify-center rounded-full bg-info text-xs text-white"
        aria-hidden="true"><Icon name="check" size={14} /></span
      >
    {/if}
    {#if card.boxes_truncated}
      <span
        class="chip pointer-events-none absolute bottom-1 left-1"
        data-testid="boxes-truncated"
        title="This image has more items than are drawn">+ more boxes</span
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
  </div>
  <div
    class="mt-auto flex items-center justify-between gap-2 px-2 pb-1.5 text-caption text-fg-2"
  >
    <span class="truncate" title={card.file_name}>{card.file_name}</span>
    <button
      type="button"
      class="shrink-0 underline hover:text-fg-1"
      data-testid="image-card-open"
      onclick={onopen}>Details</button
    >
  </div>
</li>

<style>
  .unlabeled-chip {
    border-color: var(--color-warn);
    color: var(--color-warn);
  }
</style>
