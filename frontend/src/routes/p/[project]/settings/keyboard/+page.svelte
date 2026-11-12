<script lang="ts">
  import KeymapCard from '$components/settings/KeymapCard.svelte';
  import EmptyState from '$components/ui/EmptyState.svelte';
  import PageHeader from '$components/ui/PageHeader.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { keymapAvailability } from '$stores/keymap.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });
</script>

<PageHeader
  title="Keyboard shortcuts"
  description="Applies to this project, in every browser."
/>
{#if keymapAvailability.available === false}
  <div data-testid="keymap-unavailable">
    <EmptyState
      title="Shortcuts cannot be changed here"
      description="This backend does not serve the keymap, so the built-in shortcuts apply."
    />
  </div>
{:else}
  <KeymapCard />
{/if}
