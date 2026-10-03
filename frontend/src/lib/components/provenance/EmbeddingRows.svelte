<!--
  Track B (v0.4.0 embedding state) rows for `CropMetaPanel`'s main <dl>.
  Renders <dt>/<dd> pairs, or nothing. Empty until the track lands.
-->
<script lang="ts">
  import type { Crop } from '$lib/types';

  let { crop: _crop }: { crop: Crop } = $props();
</script>
