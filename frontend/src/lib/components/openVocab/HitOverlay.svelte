<script lang="ts">
  /**
   * An uploaded test image with the served hits drawn over it: boxes and
   * outlines are already normalised to the image, so an SVG on a 0-1 view
   * box draws them as served. A dropped hit (`dimmed`) is drawn faint.
   */
  import OverlayShapesLayer from '$components/configTest/OverlayShapesLayer.svelte';
  import type { OverlayShape } from '$lib/configTest/overlayShapes';

  interface Props {
    src: string;
    shapes: OverlayShape[];
  }

  let { src, shapes }: Props = $props();
</script>

<div class="relative inline-block max-w-full" data-testid="hit-overlay">
  <img {src} alt="test upload" class="block max-h-96 max-w-full" />
  <OverlayShapesLayer {shapes} boxTestid="hit-box" polygonTestid="hit-polygon" />
</div>
