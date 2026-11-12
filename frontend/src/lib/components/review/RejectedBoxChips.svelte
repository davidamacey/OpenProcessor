<!--
  One chip per rejected box of the region item under review (W8: no separate
  "candidate" concept, a verifier-rejected box is a box with state
  'rejected' and its own rejectionReason). Styled by the served rejection
  kind, worded per box. An unsaved local box has no id yet, so the key falls
  back to its position.
-->
<script lang="ts">
  import { regionVocabularyStore } from '$stores/regionVocabulary.svelte';
  import type { EditableBox } from '$lib/annotations/multiBox';
  import type { SlotBox } from '$lib/annotations/types';

  let { boxes, subBoxes }: { boxes: EditableBox[]; subBoxes: SlotBox[] | undefined } =
    $props();

  const rejected = $derived(boxes.filter((b) => b.state === 'rejected'));
</script>

{#each rejected as b, i (b.boxId ?? `new-${i}`)}
  {@const kind = regionVocabularyStore.rejectionReasonKind(
    subBoxes?.find((sb) => sb.boxId === b.boxId)?.rejectionReason ?? null,
  )}
  <span
    class={`rounded border px-1.5 py-0.5 text-caption ${
      kind === 'needs_human'
        ? 'border-line-strong bg-surface-2/80 text-fg-2'
        : kind === 'model_verdict'
          ? 'border-danger/40 bg-danger/15 text-danger'
          : 'border-warn/40 bg-warn/15 text-warn'
    }`}
  >
    {kind === 'needs_human' ? 'candidate · needs review' : 'rejected · confirm to accept'}
  </span>
{/each}
