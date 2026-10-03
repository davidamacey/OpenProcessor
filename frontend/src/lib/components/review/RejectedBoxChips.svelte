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
    class={`rounded border px-1.5 py-0.5 text-[10px] ${
      kind === 'needs_human'
        ? 'border-zinc-600 bg-zinc-800/80 text-zinc-300'
        : kind === 'model_verdict'
          ? 'border-red-500/40 bg-red-500/15 text-red-200'
          : 'border-amber-500/40 bg-amber-500/15 text-amber-200'
    }`}
  >
    {kind === 'needs_human' ? 'candidate · needs review' : 'rejected · confirm to accept'}
  </span>
{/each}
