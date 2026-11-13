<script lang="ts">
  /**
   * The objects of the image as the server serves them: the label it decided, the tone, the lock and
   * whether the object was deleted. Clicking a row selects it. No keyboard handler of its own: Tab,
   * delete and the class picker are keymap actions.
   */
  import { boxToneClass } from '$lib/boxes/boxTone';
  import type { ObjectRow } from '$lib/editor/editorController.svelte';
  import { keymapStore } from '$stores/keymap.svelte';

  interface Props {
    rows: ObjectRow[];
    selectedCropId: string | null;
    onselect: (cropId: string) => void;
    onclass: (cropId: string) => void;
    ondelete: (cropId: string) => void;
  }

  let { rows, selectedCropId, onselect, onclass, ondelete }: Props = $props();

  const small =
    'rounded-sm border border-line-strong px-1.5 py-0.5 text-meta text-fg-1 hover:bg-surface-2 focus-visible:ring-2 focus-visible:ring-info focus-visible:outline-none';
</script>

<ul
  class="flex flex-col gap-1"
  aria-label="Objects in this image"
  data-testid="object-list"
>
  {#each rows as row (row.cropId)}
    {@const selected = row.cropId === selectedCropId}
    <li
      class="rounded-md border px-2 py-1.5 {selected
        ? 'border-info bg-info/10'
        : 'border-line'} {row.deleted ? 'opacity-60' : ''}"
      data-crop-id={row.cropId}
      data-selected={selected}
    >
      <button
        type="button"
        class="flex w-full items-center gap-2 text-left text-body text-fg-1 focus-visible:ring-2 focus-visible:ring-info focus-visible:outline-none"
        aria-current={selected ? 'true' : undefined}
        onclick={() => onselect(row.cropId)}
      >
        <span
          class="h-3 w-3 shrink-0 rounded-sm border-2 {boxToneClass(row.tone)}"
          aria-hidden="true"
        ></span>
        <span class="min-w-0 flex-1 truncate {row.deleted ? 'line-through' : ''}">
          {row.label}
        </span>
        {#if row.locked}
          <span
            class="rounded-sm border border-line-strong px-1 text-meta text-fg-2"
            title={row.lockLabel ?? 'Locked'}
            data-testid="object-locked"
          >
            {row.lockLabel ?? 'Locked'}
          </span>
        {/if}
        {#if row.regionCount > 0}
          <span class="text-meta text-fg-2">{row.regionCount} regions</span>
        {/if}
        {#if row.deleted}
          <span class="text-meta text-fg-2" data-testid="object-deleted">Deleted</span>
        {/if}
      </button>
      {#if selected && !row.deleted}
        <div class="mt-1.5 flex gap-1.5">
          <button
            type="button"
            class={small}
            title={keymapStore.glyph('editor.class_picker')}
            onclick={() => onclass(row.cropId)}
          >
            Change class
          </button>
          <button
            type="button"
            class={small}
            title={keymapStore.glyph('editor.delete')}
            onclick={() => ondelete(row.cropId)}
          >
            Delete
          </button>
        </div>
      {/if}
    </li>
  {:else}
    <li class="px-2 py-3 text-body text-fg-2" data-testid="object-list-empty">
      No objects yet. Draw one to start.
    </li>
  {/each}
</ul>
