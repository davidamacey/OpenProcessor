<script lang="ts">
  /**
   * The crop thumbnail with the profile-test candidates drawn in the
   * crop's own frame: the server's `bbox_in_parent` /
   * `mask_polygon_in_parent`, drawn as served as an SVG layer (§7.7, no
   * client projection). Dropped candidates are dimmed.
   */
  import { getThumbUrl } from '$lib/api';
  import type { OverlayShape } from '$lib/configTest/overlayShapes';

  interface Props {
    cropId: string;
    shapes: OverlayShape[];
  }

  let { cropId, shapes }: Props = $props();
</script>

<div class="relative inline-block" data-testid="crop-frame-shapes">
  <img
    src={getThumbUrl(cropId, 256)}
    alt="crop {cropId}"
    class="block max-h-64 max-w-full"
  />
  <svg
    class="absolute inset-0 h-full w-full"
    viewBox="0 0 1 1"
    preserveAspectRatio="none"
    aria-hidden="true"
  >
    {#each shapes as s (s.key)}
      {#if s.kind === 'box'}
        <rect
          x={s.box[0]}
          y={s.box[1]}
          width={s.box[2] - s.box[0]}
          height={s.box[3] - s.box[1]}
          fill="none"
          stroke="rgb(56, 189, 248)"
          stroke-width="2"
          stroke-dasharray={s.dimmed ? '4 3' : undefined}
          opacity={s.dimmed ? 0.4 : 1}
          vector-effect="non-scaling-stroke"
          data-testid="crop-frame-box"
          data-dimmed={s.dimmed}><title>{s.title}</title></rect
        >
      {:else}
        <polygon
          points={s.points.map((q) => `${q[0]},${q[1]}`).join(' ')}
          fill="rgba(56, 189, 248, 0.15)"
          stroke="rgb(56, 189, 248)"
          stroke-width="2"
          stroke-dasharray={s.dimmed ? '4 3' : undefined}
          opacity={s.dimmed ? 0.4 : 1}
          vector-effect="non-scaling-stroke"
          data-testid="crop-frame-polygon"
          data-dimmed={s.dimmed}><title>{s.title}</title></polygon
        >
      {/if}
    {/each}
  </svg>
</div>
