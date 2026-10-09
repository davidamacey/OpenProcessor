<!--
  The detector's own class (and score) beside a VLM-sourced label, so the
  two answers can be compared. Renders nothing for any other label source
  or an item that carries no detector answer.
-->
<script lang="ts">
  import { detectorClassText } from '$lib/labelConfirmation';
  import { classSourcesStore } from '$stores/classSources.svelte';
  import type { Crop } from '$lib/types';

  interface Props {
    crop: Pick<Crop, 'class_source' | 'detector_class_name' | 'detector_confidence'>;
    class?: string;
    /** Value only, for a row that already names it "Detector class". */
    bare?: boolean;
  }
  let { crop, class: cls = '', bare = false }: Props = $props();

  const text = $derived(
    detectorClassText(classSourcesStore.roleFor(crop.class_source), crop),
  );
</script>

{#if text}
  <span
    class="min-w-0 truncate {cls}"
    title="The detector's own class for this item"
    data-testid="detector-class"
  >
    {#if !bare}Detector:&nbsp;{/if}<span class="text-zinc-200">{text}</span>
  </span>
{/if}
