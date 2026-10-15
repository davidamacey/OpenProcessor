<script lang="ts">
  /**
   * The VLM's suggestions for an item or one box: pack-declared attributes
   * as chips with the VLM's confidence, and, when its class answer differs
   * from the item's class (a locked label keeps its class), "VLM said X".
   * Dashed and labelled "VLM suggests" so it never reads as a label; the
   * values are served (`assist`), nothing here is derived.
   */
  import type { ItemAssist } from '$lib/types_assist';

  interface Props {
    assist: ItemAssist | null | undefined;
    /** The item's current class, to tell when the VLM disagrees. */
    className?: string | null;
    /** Show this box's attributes instead of the item's. */
    boxId?: string | null;
  }

  let { assist, className = null, boxId = null }: Props = $props();

  const attributes = $derived(
    boxId
      ? (assist?.boxes?.find((b) => b.box_id === boxId)?.attributes ?? [])
      : (assist?.attributes ?? []),
  );
  const said = $derived(
    !boxId && assist?.class_name && assist.class_name !== className
      ? assist.class_name
      : null,
  );
</script>

{#if attributes.length > 0 || said}
  <div
    class="flex flex-wrap items-center gap-1 text-[11px]"
    data-testid="assist-suggestions"
    title="Suggestions from the VLM, not labels. Nothing here changes the item's class or boxes."
  >
    <span class="text-[10px] uppercase tracking-wide text-zinc-500">VLM suggests</span>
    {#if said}
      <span
        class="rounded border border-dashed border-amber-500/50 px-1 text-amber-200"
        data-testid="assist-said"
      >
        VLM said: {said}{assist?.class_confidence ? ` (${assist.class_confidence})` : ''}
      </span>
    {/if}
    {#each attributes as a (a.key)}
      <span
        class="inline-flex items-center gap-1 rounded border border-dashed border-sky-500/50 px-1 text-sky-200"
        data-testid="assist-chip"
        data-confidence={a.confidence}
      >
        <span class="text-sky-300/70">{a.key}:</span>
        <span>{a.value}</span>
        <span class="text-[10px] text-zinc-500">{a.confidence}</span>
      </span>
    {/each}
  </div>
{/if}
