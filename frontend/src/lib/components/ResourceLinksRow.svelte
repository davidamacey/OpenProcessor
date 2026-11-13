<script lang="ts">
  /**
   * The "Dashboards" menu in the header of /train and /bakeoff: the served `service` entries
   * of `GET /settings` `resource_links`, in served order, through the same item rendering as
   * the Resources menu. Docs entries and the client-owned Documentation entry belong to that
   * menu only. Nothing is guessed: no served service entry, no menu.
   */
  import { onMount } from 'svelte';
  import ResourceLinkItem from '$components/ResourceLinkItem.svelte';
  import Icon from '$components/ui/Icon.svelte';
  import Popover from '$components/ui/Popover.svelte';
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
  <div class="shrink-0" data-testid="resource-row">
    <Popover label="Dashboards" align="end" panelClass="w-64 py-1">
      {#snippet trigger({ open, attrs })}
        <button type="button" class="btn" data-testid="dashboards-trigger" {...attrs}>
          Dashboards
          <Icon name={open ? 'chevron-up' : 'chevron-down'} size={14} />
        </button>
      {/snippet}
      <div data-testid="dashboards-list">
        {#each links as l (l.id)}
          <ResourceLinkItem view={l} class="px-3 py-1.5 text-body hover:bg-surface-2" />
        {/each}
      </div>
    </Popover>
  </div>
{/if}
