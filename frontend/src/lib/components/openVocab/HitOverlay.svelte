<script lang="ts">
  /**
   * An uploaded test image with the served hits drawn over it: boxes and
   * outlines are already normalised to the image, so an SVG on a 0-1 view
   * box draws them as served. A dropped hit (`dimmed`) is drawn faint.
   */
  import type { OverlayShape } from '$lib/configTest/overlayShapes';

  interface Props {
    src: string;
    shapes: OverlayShape[];
  }

  let { src, shapes }: Props = $props();
</script>

<div class="relative inline-block max-w-full" data-testid="hit-overlay">
  <img {src} alt="test upload" class="block max-h-96 max-w-full" />
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
          data-testid="hit-box"
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
          data-testid="hit-polygon"
          data-dimmed={s.dimmed}><title>{s.title}</title></polygon
        >
      {/if}
    {/each}
  </svg>
</div>
