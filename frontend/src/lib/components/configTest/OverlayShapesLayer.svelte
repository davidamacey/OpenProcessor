<script lang="ts">
  /**
   * The SVG layer of served overlay shapes on a 0-1 view box: boxes and
   * outlines are drawn as served, a dropped shape (`dimmed`) faint. Sits
   * absolutely over the image its parent wraps.
   */
  import type { OverlayShape } from '$lib/configTest/overlayShapes';

  interface Props {
    shapes: OverlayShape[];
    boxTestid: string;
    polygonTestid: string;
  }

  let { shapes, boxTestid, polygonTestid }: Props = $props();
</script>

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
        data-testid={boxTestid}
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
        data-testid={polygonTestid}
        data-dimmed={s.dimmed}><title>{s.title}</title></polygon
      >
    {/if}
  {/each}
</svg>
