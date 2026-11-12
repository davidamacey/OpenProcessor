<script lang="ts">
  import KeymapCard from '$components/settings/KeymapCard.svelte';
  import PageHeader from '$components/ui/PageHeader.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });
</script>

<PageHeader
  title="Keyboard shortcuts"
  description="Applies to this project, in every browser."
/>
<KeymapCard />
