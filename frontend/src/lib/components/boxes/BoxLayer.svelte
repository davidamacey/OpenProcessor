<script lang="ts">
  /**
   * Draws the boxes the backend served for an image (`BoxView`: image coordinates, label, tone),
   * and nothing else: no label, colour or geometry is derived here. The parent supplies the
   * positioned wrapper; boxes are placed in percent of it.
   */
  import type { ImageBox } from '$lib/types';
  import { boxToneClass } from '$lib/boxes/boxTone';
  import { regionStatusesStore } from '$stores/regionStatuses.svelte';

  interface Props {
    boxes: ImageBox[];
    /** The item drawn with a ring; its box is not clickable. */
    selectedCropId?: string | null;
    /** Fires for a click on a box of another item. Absent = boxes are not clickable. */
    onselect?: (cropId: string) => void;
  }

  let { boxes, selectedCropId = null, onselect }: Props = $props();

  function pct(n: number): string {
    return `${(n * 100).toFixed(3)}%`;
  }

  function dashed(b: ImageBox): boolean {
    return (
      b.state != null && (regionStatusesStore.boxStateInfo(b.state)?.dashed ?? false)
    );
  }
</script>

{#each boxes as b, i (`${b.kind}-${b.crop_id}-${b.box_id ?? ''}-${i}`)}
  {@const [x1, y1, x2, y2] = b.bbox_norm}
  {@const selected = b.kind === 'item' && b.crop_id === selectedCropId}
  {@const clickable = b.kind === 'item' && !selected && !!onselect}
  {@const boxClass = `absolute border-2 ${boxToneClass(b.tone)} ${dashed(b) ? 'border-dashed' : ''} ${
    selected ? 'z-10 ring-2 ring-white/80' : b.kind === 'item' ? 'opacity-60' : ''
  }`}
  {@const boxStyle = `left:${pct(x1)}; top:${pct(y1)}; width:${pct(x2 - x1)}; height:${pct(y2 - y1)};`}
  {#snippet label()}
    {#if selected || b.kind !== 'item'}
      <span
        class="pointer-events-none absolute -top-4 left-0 rounded-sm bg-zinc-950/90 px-1 py-0.5 text-[9px] leading-none whitespace-nowrap text-zinc-100"
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
