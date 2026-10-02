<!--
  Track C (W10 import and lock) provenance rows for `CropMetaPanel`'s main <dl>.
  Renders <dt>/<dd> pairs, or nothing. Empty until the track lands.
-->
<script lang="ts">
  import type { Crop } from '$lib/types';

  let { crop: _crop }: { crop: Crop } = $props();
</script>
