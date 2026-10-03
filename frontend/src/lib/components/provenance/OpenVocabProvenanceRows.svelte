<!--
  Track A (v0.4.0 open vocabulary: source prompt, set and revision, region
  gate skip) provenance rows for `CropMetaPanel`'s main <dl>. Renders
  <dt>/<dd> pairs, or nothing. Empty until the track lands.
-->
<script lang="ts">
  import type { Crop } from '$lib/types';

  let { crop: _crop }: { crop: Crop } = $props();
</script>
