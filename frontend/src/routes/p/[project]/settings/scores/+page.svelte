<script lang="ts">
  import ScoresCard from '$components/ScoresCard.svelte';
  import PageHeader from '$components/ui/PageHeader.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });
</script>

<PageHeader
  title="Curation scores"
  description="How much of the project each scorer covers, and a run to compute the rest."
/>
<ScoresCard />
