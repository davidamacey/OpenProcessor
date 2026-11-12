<!--
  The served page of items around the focused one, as a row of thumbnails, with where the focus
  sits in the whole served order ("12 of 412"). Shared by the Review view and the image editor:
  the caller loads the page (it is the Workspace query's own order), this only draws it.
-->
<script lang="ts">
  import { getThumbUrl } from '$lib/api_cropRead';

  interface Props {
    /** The served page, in the served order. */
    items: { id: string; displayName: string }[];
    focusedId: string | null;
    /** The served 0-based index of the focused item and the served total. */
    index: number | null;
    total: number | null;
    onSelect: (id: string) => void;
  }

  let { items, focusedId, index, total, onSelect }: Props = $props();

  // A thumbnail the server cannot serve (the source file is gone) leaves an empty tile, not a
  // broken-image icon.
  let failed = $state<Record<string, boolean>>({});

  const position = $derived(
    index !== null && total !== null ? `${index + 1} of ${total}` : null,
  );
</script>

<nav
  class="flex items-center gap-3 border-t border-zinc-800 px-3 py-2"
  aria-label="Filmstrip"
>
  {#if position}
    <span class="shrink-0 text-xs text-zinc-400" data-testid="filmstrip-position">
      {position}
    </span>
  {/if}
  <ul class="flex min-w-0 gap-1 overflow-x-auto">
    {#each items as item (item.id)}
      <li class="shrink-0">
        <button
          type="button"
          class="block h-14 w-14 overflow-hidden rounded border {item.id === focusedId
            ? 'border-blue-500 ring-1 ring-blue-500'
            : 'border-zinc-700 hover:border-zinc-500'}"
          aria-current={item.id === focusedId ? 'true' : undefined}
          aria-label={item.displayName}
          data-testid="filmstrip-item"
          onclick={() => onSelect(item.id)}
        >
          {#if !failed[item.id]}
            <img
              src={getThumbUrl(item.id, 112)}
              alt=""
              class="h-full w-full object-cover"
              loading="lazy"
              onerror={() => (failed[item.id] = true)}
            />
          {/if}
        </button>
      </li>
    {/each}
  </ul>
</nav>
