<!--
  The /review quick-assign class strip: the item's most likely classes, each
  with its bound hotkey, plus the button that opens the class picker.
-->
<script lang="ts">
  import { kg } from '$lib/review/reviewKeys';
  import type { RegistryClass } from '$lib/types_classes';

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
      class="rounded border border-line-strong bg-surface-1 px-2 py-1 text-meta text-fg-1
         hover:border-accent/60 hover:bg-accent/10 hover:text-white
         focus:outline-none focus:ring-2 focus:ring-accent/40"
      title={cls.hotkey_letter
        ? `Assign ${cls.name} (press ${cls.hotkey_letter})`
        : `Assign ${cls.name}`}
      onclick={() => assign(cls.id)}
    >
      {#if cls.hotkey_letter}
        <kbd
          class="mr-1.5 rounded bg-surface-2 px-1 py-0.5 text-caption text-accent-text"
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
    class="rounded border border-dashed border-line-strong bg-surface-1 px-2 py-1 text-meta text-fg-2
       hover:border-accent/60 hover:bg-accent/10 hover:text-white
       focus:outline-none focus:ring-2 focus:ring-accent/40"
    title="Search all classes ({kg('review.queue.class_picker')})"
    onclick={openPicker}
  >
    <kbd class="mr-1.5 rounded bg-surface-2 px-1 py-0.5 text-caption text-accent-text">
      {kg('review.queue.class_picker')}
    </kbd>
    search all classes…
  </button>
</div>
<p class="mt-1.5 text-caption text-fg-3">
  Click a class, press its bound letter, or press {kg('review.queue.class_picker')} to search
  all classes (set hotkeys on /classes).
</p>
