<script lang="ts">
  /**
   * Draws the boxes the backend served for an image (`BoxView`: image coordinates, label, tone),
   * and nothing else: no label, colour or geometry is derived here. The parent supplies the
   * positioned wrapper; boxes are placed in percent of it.
   */
  import type { ImageBox } from '$lib/types_images';
  import { boxToneClass } from '$lib/boxes/boxTone';
  import { regionStatusesStore } from '$stores/regionStatuses.svelte';

  interface Props {
    boxes: ImageBox[];
    /** The item drawn with a ring; its box is not clickable. */
    selectedCropId?: string | null;
    /** Fires for a click on a box of another item. Absent = boxes are not clickable. */
    onselect?: (cropId: string) => void;
    /** False on a thumbnail too small for text: boxes draw, labels do not. */
    showLabels?: boolean;
    /** Region boxes of every item. Off, only the selected item's region boxes draw. */
    allRegions?: boolean;
    /** The selected box is the subject: 3 px in the highlight with everything outside it dimmed
     *  (the parent clips the dim), the other boxes thin, faint and unlabelled. */
    spotlight?: boolean;
  }

  let {
    boxes,
    selectedCropId = null,
    onselect,
    showLabels = true,
    allRegions = false,
    spotlight = false,
  }: Props = $props();

  const visible = $derived(
    boxes.filter(
      (b) => b.kind !== 'region' || allRegions || b.crop_id === selectedCropId,
    ),
  );

  function pct(n: number): string {
    return `${(n * 100).toFixed(3)}%`;
  }

  function dashed(b: ImageBox): boolean {
    return (
      b.state != null && (regionStatusesStore.boxStateInfo(b.state)?.dashed ?? false)
    );
  }
</script>

{#each visible as b, i (`${b.kind}-${b.crop_id}-${b.box_id ?? ''}-${i}`)}
  {@const [x1, y1, x2, y2] = b.bbox_norm}
  {@const selected = b.kind === 'item' && b.crop_id === selectedCropId}
  {@const clickable = b.kind === 'item' && !selected && !!onselect}
  {@const boxClass = `absolute ${
    spotlight && selected
      ? 'z-10 border-3 border-hl dim-outside'
      : `${spotlight ? 'border opacity-40' : 'border-2'} ${boxToneClass(b.tone, b.label_kind)} ${
          !spotlight && b.kind === 'item' && !selected ? 'opacity-60' : ''
        } ${selected ? 'z-10 ring-2 ring-white/80' : ''}`
  } ${dashed(b) ? 'border-dashed' : ''}`}
  {@const boxStyle = `left:${pct(x1)}; top:${pct(y1)}; width:${pct(x2 - x1)}; height:${pct(y2 - y1)};`}
  {#snippet label()}
    {#if showLabels && (!spotlight || selected)}
      <span
        class="pointer-events-none absolute -top-4 left-0 rounded-sm bg-surface-0/90 px-1 py-0.5 text-caption leading-none whitespace-nowrap text-fg-1"
      >
        {b.label}
      </span>
    {/if}
  {/snippet}
  {#if clickable}
    <button
      type="button"
      class="pointer-events-auto cursor-pointer {boxClass}"
      style={boxStyle}
      data-testid="overlay-box"
      data-kind={b.kind}
      data-crop-id={b.crop_id}
      data-tone={b.tone}
      title={b.label}
      onclick={() => onselect?.(b.crop_id)}
    >
      {@render label()}
    </button>
  {:else}
    <div
      class={boxClass}
      style={boxStyle}
      data-testid="overlay-box"
      data-kind={b.kind}
      data-crop-id={b.crop_id}
      data-tone={b.tone}
      title={b.label}
    >
      {@render label()}
    </div>
  {/if}
{/each}

<style>
  /* The shadow spreads far past the box; the parent clips it to the image. */
  :global(.dim-outside) {
    box-shadow: 0 0 0 100vmax rgb(0 0 0 / 0.55);
  }
</style>
