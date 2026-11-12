<!--
  A row of thumbnails around the focused one, with where the focus sits in the served order
  ("12 of 412"). Shared by Review (buttons) and the image editor (links): the caller supplies the
  served page, in the served order; this only draws it. A tile whose thumbnail cannot be served
  (the source file is gone) stays an empty tile, not a broken-image icon.
-->
<script lang="ts">
  import { resolve } from '$app/paths';
  import ScrollStrip from '$components/ScrollStrip.svelte';

  interface FilmstripTile {
    id: string;
    /** The tile's accessible name. */
    label: string;
    thumbUrl: string;
    /** Set: the tile is a link to it; unset: a button that calls `onSelect`. */
    href?: string;
  }

  interface Props {
    tiles: FilmstripTile[];
    focusedId: string | null;
    /** The served 0-based index of the focused item and the served total. */
    index?: number | null;
    total?: number | null;
    navLabel?: string;
    onSelect?: (id: string) => void;
  }

  let {
    tiles,
    focusedId,
    index = null,
    total = null,
    navLabel = 'Filmstrip',
    onSelect,
  }: Props = $props();

  let failed = $state<Record<string, boolean>>({});

  const position = $derived(
    index !== null && total !== null ? `${index + 1} of ${total}` : null,
  );
  const TILE =
    'block h-14 w-20 shrink-0 overflow-hidden rounded-md border-2 focus-visible:ring-2 focus-visible:ring-info focus-visible:outline-none';
  const tone = (current: boolean) =>
    current ? 'border-info' : 'border-line opacity-80 hover:opacity-100';
</script>

{#snippet thumb(tile: FilmstripTile)}
  {#if !failed[tile.id]}
    <img
      src={tile.thumbUrl}
      alt=""
      loading="lazy"
      class="h-full w-full object-cover"
      onerror={() => (failed[tile.id] = true)}
    />
  {/if}
{/snippet}

<div class="flex items-center gap-3">
  {#if position}
    <span class="shrink-0 text-xs text-fg-2" data-testid="filmstrip-position">
      {position}
    </span>
  {/if}
  <ScrollStrip {navLabel} activeKey={focusedId} class="gap-2" testId="filmstrip">
    {#each tiles as tile (tile.id)}
      {@const current = tile.id === focusedId}
      {#if tile.href}
        <a
          href={resolve(tile.href)}
          class="{TILE} {tone(current)}"
          aria-current={current ? 'true' : undefined}
          data-active={current ? 'true' : undefined}
          aria-label={tile.label}
          data-testid="filmstrip-item"
          data-image-id={tile.id}
        >
          {@render thumb(tile)}
        </a>
      {:else}
        <button
          type="button"
          class="{TILE} {tone(current)}"
          aria-current={current ? 'true' : undefined}
          data-active={current ? 'true' : undefined}
          aria-label={tile.label}
          data-testid="filmstrip-item"
          onclick={() => onSelect?.(tile.id)}
        >
          {@render thumb(tile)}
        </button>
      {/if}
    {/each}
  </ScrollStrip>
</div>
