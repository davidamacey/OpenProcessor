<script lang="ts">
  import Icon from '$components/ui/Icon.svelte';
  /**
   * The thumbnails of the crops selected on the scatter, so a selection of dots can be judged
   * before it is acted on. Capped (a huge lasso must not render thousands of <img> tags); a
   * click opens one crop enlarged. Built from points the plot already holds, no new fetch.
   */
  import { getThumbUrl } from '$lib/api_cropRead';
  import type { VizPoint } from '$lib/api_viz';
  import Modal from '$components/Modal.svelte';

  interface Props {
    /** The selected points, in the plot's order. */
    points: VizPoint[];
    /** How many are selected (may exceed `points` when some are not on the scatter). */
    count: number;
    height: number;
  }

  let { points, count, height }: Props = $props();

  const SELECTED_PREVIEW_CAP = 60;
  const selectedPreview = $derived(points.slice(0, SELECTED_PREVIEW_CAP));
  let expandedPreviewId = $state<string | null>(null);
  const expandedPoint = $derived(points.find((p) => p.crop_id === expandedPreviewId));
</script>

<div
  class="flex w-40 shrink-0 flex-col gap-1 overflow-y-auto rounded-md border border-line bg-surface-0 p-2"
  style:height="{height}px"
  data-testid="embedding-preview"
>
  <p class="text-caption text-fg-3">{count} selected, click to enlarge</p>
  <div class="grid grid-cols-2 gap-1">
    {#each selectedPreview as p (p.crop_id)}
      <button
        type="button"
        class="group relative aspect-square overflow-hidden rounded border border-line-strong hover:border-accent"
        title={p.class_name ?? 'unlabeled'}
        onclick={() => (expandedPreviewId = p.crop_id)}
      >
        <img
          src={getThumbUrl(p.crop_id, 64)}
          alt="crop {p.crop_id}"
          draggable="false"
          class="h-full w-full object-cover [-webkit-user-drag:none]"
        />
        {#if p.class_name}
          <span
            class="absolute inset-x-0 bottom-0 truncate bg-black/70 px-1 text-center text-caption text-fg-1"
          >
            {p.class_name}
          </span>
        {/if}
      </button>
    {/each}
  </div>
  {#if count > SELECTED_PREVIEW_CAP}
    <p class="text-center text-caption text-fg-3">
      +{count - SELECTED_PREVIEW_CAP} more
    </p>
  {/if}
</div>

{#if expandedPreviewId}
  <Modal
    label="Selected crop preview"
    onclose={() => (expandedPreviewId = null)}
    layer="top"
    scrim="bg-black/80"
    panelClass="max-h-full max-w-2xl"
  >
    <img
      src={getThumbUrl(expandedPreviewId, 512)}
      alt="crop {expandedPreviewId}"
      draggable="false"
      class="max-h-[80vh] max-w-full rounded-md border border-line-strong [-webkit-user-drag:none]"
    />
    {#if expandedPoint?.class_name}
      <p
        class="absolute inset-x-0 bottom-0 bg-black/70 px-2 py-1 text-center text-body text-fg-1"
      >
        {expandedPoint.class_name}
      </p>
    {/if}
    <button
      type="button"
      class="absolute -top-3 -right-3 rounded-full border border-line-strong bg-surface-1 px-2 py-1 text-body text-white"
      onclick={(e) => {
        e.stopPropagation();
        expandedPreviewId = null;
      }}
      aria-label="Close"
    >
      <Icon name="x" size={14} />
    </button>
  </Modal>
{/if}
