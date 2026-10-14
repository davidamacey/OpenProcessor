<!--
  The /review quick-assign class strip: the item's most likely classes, each
  with its bound hotkey, plus the button that opens the class picker.
-->
<script lang="ts">
  import { kg } from '$lib/review/reviewKeys';
  import type { RegistryClass } from '$lib/types';

  interface Props {
    topClasses: RegistryClass[];
    assign: (classId: number) => Promise<void>;
    openPicker: () => void;
  }

  let { topClasses, assign, openPicker }: Props = $props();
</script>

<div class="mt-3 flex flex-wrap gap-1.5">
  {#each topClasses as cls (cls.id)}
    <button
      type="button"
      class="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200
         hover:border-blue-500/60 hover:bg-blue-500/10 hover:text-white
         focus:outline-none focus:ring-2 focus:ring-blue-500/40"
      title={cls.hotkey_letter
        ? `Assign ${cls.name} (press ${cls.hotkey_letter})`
        : `Assign ${cls.name}`}
      onclick={() => assign(cls.id)}
    >
      {#if cls.hotkey_letter}
        <kbd
          class="mr-1.5 rounded bg-zinc-800 px-1 py-0.5 font-mono text-[10px] uppercase text-blue-300"
        >
          {cls.hotkey_letter}
        </kbd>
      {/if}
      {cls.name}
    </button>
  {/each}
  <!-- P1-4: only the 10 most-validated classes are one click above;
     this opens the fuzzy-search picker over all non-deprecated
     classes (same action as pressing /). -->
  <button
    type="button"
    class="rounded border border-dashed border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-400
       hover:border-blue-500/60 hover:bg-blue-500/10 hover:text-white
       focus:outline-none focus:ring-2 focus:ring-blue-500/40"
    title="Search all classes ({kg('review.queue.class_picker')})"
    onclick={openPicker}
  >
    <kbd
      class="mr-1.5 rounded bg-zinc-800 px-1 py-0.5 font-mono text-[10px] text-blue-300"
    >
      {kg('review.queue.class_picker')}
    </kbd>
    search all classes…
  </button>
</div>
<p class="mt-1.5 text-[10px] text-zinc-500">
  Click a class, press its bound letter, or press {kg('review.queue.class_picker')} to search
  all classes (set hotkeys on /classes).
</p>
