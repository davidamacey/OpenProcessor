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
    class="shrink-0 rounded-sm border border-line-strong bg-surface-2 px-1 py-0.5 text-caption text-fg-1"
    data-testid="confirmation-chip">{text}</span
  >
{/if}
