<!--
  "Human-confirmed" / "VLM suggestion" / "Auto-validated": what a class
  label is worth as ground truth, from the served source role and the
  validated flag. Renders nothing for a role that has no such wording.
-->
<script lang="ts">
  import { confirmationLabel } from '$lib/sourceBadge';
  import { classSourcesStore } from '$stores/classSources.svelte';

  interface Props {
    source: string | null | undefined;
    validated: boolean;
  }
  let { source, validated }: Props = $props();

  const text = $derived(confirmationLabel(classSourcesStore.roleFor(source), validated));
</script>

{#if text}
  <span
    class="shrink-0 rounded-sm border border-zinc-600 bg-zinc-800 px-1 py-0.5 text-[10px] text-zinc-200"
    data-testid="confirmation-chip">{text}</span
  >
{/if}
