<script lang="ts">
  /**
   * The crop thumbnail with the profile-test candidates drawn in the
   * crop's own frame: the server's `bbox_in_parent` /
   * `mask_polygon_in_parent`, drawn as served as an SVG layer (§7.7, no
   * client projection). Dropped candidates are dimmed.
   */
  import { getThumbUrl } from '$lib/api_cropRead';
  import OverlayShapesLayer from '$components/configTest/OverlayShapesLayer.svelte';
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
  <OverlayShapesLayer
    {shapes}
    boxTestid="crop-frame-box"
    polygonTestid="crop-frame-polygon"
  />
</div>
