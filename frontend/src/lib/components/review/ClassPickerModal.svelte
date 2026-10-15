<!--
  The /review class picker dialog: fuzzy search over every assignable class.
-->
<script lang="ts">
  import Modal from '$components/Modal.svelte';
  import type { ReviewClassPicker } from '$lib/review/reviewClassPicker.svelte';

  let { picker }: { picker: ReviewClassPicker } = $props();

  const pickerOpen = $derived(picker.pickerOpen);
  const pickerClasses = $derived(picker.pickerClasses);
  const pickerResults = $derived(picker.pickerResults);
  const pickerIndex = $derived(picker.pickerIndex);
  const closePicker = $derived(picker.closePicker);
  const pickClass = $derived(picker.pickClass);
  const onPickerKeydown = $derived(picker.onPickerKeydown);
</script>

{#if pickerOpen}
  <!-- Class picker (P1-4) — fuzzy-search over every non-deprecated class,
       opened with / or the "search all classes…" button. Backdrop click
       and Esc both close without assigning. -->
  <Modal
    label="Search classes"
    onclose={closePicker}
    layer="top"
    align="top"
    scrim="bg-black/50"
    panelClass="w-full max-w-md overflow-hidden rounded-lg border border-zinc-700 bg-zinc-900 shadow-xl"
  >
    <input
      bind:this={picker.pickerInputEl}
      bind:value={picker.pickerQuery}
      type="text"
      placeholder="Search all {pickerClasses.length} classes…"
      class="w-full border-b border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:outline-none"
      onkeydown={onPickerKeydown}
    />
    <ul class="max-h-72 overflow-y-auto py-1 text-sm">
      {#if pickerResults.length === 0}
        <li class="px-3 py-2 text-zinc-500">No matching class.</li>
      {/if}
      {#each pickerResults as cls, i (cls.id)}
        <li>
          <button
            type="button"
            class="flex w-full items-center gap-2 px-3 py-1.5 text-left {i === pickerIndex
              ? 'bg-blue-500/20 text-white'
              : 'text-zinc-200 hover:bg-zinc-800'}"
            onclick={() => pickClass(cls)}
            onmouseenter={() => (picker.pickerIndex = i)}
          >
            {#if cls.hotkey_letter}
              <kbd
                class="rounded bg-zinc-800 px-1 py-0.5 font-mono text-[10px] uppercase text-blue-300"
              >
                {cls.hotkey_letter}
              </kbd>
            {/if}
            <span class="grow truncate">{cls.name}</span>
            <span class="shrink-0 text-[10px] text-zinc-500"
              >{cls.validated_count} validated</span
            >
          </button>
        </li>
      {/each}
    </ul>
    <p class="border-t border-zinc-800 px-3 py-1.5 text-[10px] text-zinc-500">
      <kbd>↑↓</kbd> navigate · <kbd>Enter</kbd> assign · <kbd>Esc</kbd> close
    </p>
  </Modal>
{/if}
