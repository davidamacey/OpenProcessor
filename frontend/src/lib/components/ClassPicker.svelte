<script lang="ts">
  /**
   * Assign or create a class. A combobox over the served class registry:
   * typed text filters the list, the last row creates `Create "<typed>"`, and
   * a refused name offers the server's suggestion. Deprecated classes are
   * never offered.
   */
  import Modal from '$components/Modal.svelte';
  import { createClassPicker } from '$lib/classes/classPicker.svelte';
  import { keymapStore } from '$stores/keymap.svelte';
  import type { RegistryClass } from '$lib/types_classes';

  interface Props {
    open: boolean;
    mode: 'assign' | 'create';
    /** Class names used last, ranked first. */
    recent?: string[];
    onpick: (cls: RegistryClass) => void;
    onclose: () => void;
  }

  let { open, mode, recent = [], onpick, onclose }: Props = $props();

  const picker = createClassPicker({
    mode: () => mode,
    recent: () => recent,
    onpick: (c) => onpick(c),
    onclose: () => onclose(),
  });
  let input = $state<HTMLInputElement | null>(null);

  $effect(() => {
    if (!open) return;
    picker.reset();
    queueMicrotask(() => input?.focus());
  });

  const listId = 'class-picker-list';
  const rowCount = $derived(picker.results.length + (picker.createName ? 1 : 0));
</script>

{#if open}
  <Modal
    label={mode === 'create' ? 'New class' : 'Search classes'}
    {onclose}
    layer="top"
    align="top"
    scrim="bg-black/50"
    panelClass="w-full max-w-md overflow-hidden rounded-lg border border-zinc-700 bg-zinc-900 shadow-xl"
  >
    <input
      bind:this={input}
      bind:value={picker.query}
      type="text"
      role="combobox"
      aria-expanded="true"
      aria-controls={listId}
      aria-activedescendant={rowCount > 0
        ? `class-picker-row-${picker.index}`
        : undefined}
      aria-label={mode === 'create' ? 'New class name' : 'Search classes'}
      placeholder={mode === 'create'
        ? 'Name the new class'
        : `Search all ${picker.total} classes…`}
      class="input w-full rounded-none border-0 border-b border-zinc-800 bg-zinc-900"
      onkeydown={picker.onKeydown}
    />
    <ul id={listId} role="listbox" class="max-h-72 overflow-y-auto py-1 text-sm">
      {#if rowCount === 0}
        <li class="px-3 py-2 text-zinc-500" role="presentation">No matching class.</li>
      {/if}
      {#each picker.results as cls, i (cls.id)}
        <li role="presentation">
          <button
            type="button"
            id="class-picker-row-{i}"
            role="option"
            aria-selected={i === picker.index}
            class="flex w-full items-center gap-2 px-3 py-1.5 text-left {i ===
            picker.index
              ? 'bg-blue-500/20 text-white'
              : 'text-zinc-200 hover:bg-zinc-800'}"
            onclick={() => picker.pick(cls)}
            onmouseenter={() => (picker.index = i)}
          >
            {#if cls.hotkey_letter}
              <kbd
                class="rounded bg-zinc-800 px-1 py-0.5 font-mono text-[10px] uppercase text-blue-300"
                >{cls.hotkey_letter}</kbd
              >
            {/if}
            <span class="grow truncate">{cls.name}</span>
            <span class="shrink-0 text-[10px] text-zinc-300">{cls.count}</span>
          </button>
        </li>
      {/each}
      {#if picker.createName}
        {@const i = picker.results.length}
        <li role="presentation">
          <button
            type="button"
            id="class-picker-row-{i}"
            role="option"
            aria-selected={i === picker.index}
            disabled={picker.busy}
            class="flex w-full items-center gap-2 px-3 py-1.5 text-left {i ===
            picker.index
              ? 'bg-blue-500/20 text-white'
              : 'text-zinc-200 hover:bg-zinc-800'}"
            onclick={() => void picker.create()}
            onmouseenter={() => (picker.index = i)}
            data-testid="class-picker-create"
          >
            <span class="grow truncate">Create "{picker.createName}"</span>
          </button>
        </li>
      {/if}
    </ul>
    {#if picker.error}
      <p class="border-t border-zinc-800 px-3 py-1.5 text-xs text-red-300" role="alert">
        {picker.error}
      </p>
    {/if}
    <p class="border-t border-zinc-800 px-3 py-1.5 text-[10px] text-zinc-500">
      Move with the arrow keys · <kbd>{keymapStore.glyph('global.close_overlay')}</kbd> closes
    </p>
  </Modal>
{/if}
