<script lang="ts">
  /**
   * "Dashboards:" row on /train and /bakeoff: the served `service` entries
   * of `GET /settings` `resource_links`, in served order, through the same
   * item rendering as the Resources menu. Docs entries and the client-owned
   * Documentation entry belong to the menu only. Nothing is guessed: no
   * served service entry, no row.
   */
  import { onMount } from 'svelte';
  import ResourceLinkItem from '$components/ResourceLinkItem.svelte';
  import { resourceViews } from '$lib/resourceLinks';
  import { curationSettingsStore } from '$stores/curationSettings.svelte';

  const links = $derived(
    resourceViews(curationSettingsStore.settings.resource_links).filter(
      (v) => v.kind === 'service',
    ),
  );

  onMount(() => {
    void curationSettingsStore.init();
  });
</script>

{#if links.length}
  <div
    class="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-zinc-400"
    data-testid="resource-row"
  >
    <span class="text-zinc-500">Dashboards:</span>
    {#each links as l (l.id)}
      <ResourceLinkItem view={l} />
    {/each}
  </div>
{/if}
