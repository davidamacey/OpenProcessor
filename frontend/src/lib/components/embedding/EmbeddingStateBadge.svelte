<!--
  Track B (v0.4.0 embedding state): a badge for an item's served
  `embedding_state`. `compact` is the `CropCard` chip form. Empty until the
  track lands.
-->
<script lang="ts">
  import type { EmbeddingState } from '$lib/types_itemFilter';

  let {
    state: _state,
    compact: _compact = false,
  }: { state: EmbeddingState | null; compact?: boolean } = $props();
</script>
