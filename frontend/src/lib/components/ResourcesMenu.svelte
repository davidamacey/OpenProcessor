<script lang="ts">
  /**
   * Top-bar "Resources" menu: the bundled docs (client-owned) then exactly
   * the served `resource_links`. Plain external anchors, so no resolve().
   */
  import { onMount } from 'svelte';
  import ResourceLinkItem from '$components/ResourceLinkItem.svelte';
  import Icon from '$components/ui/Icon.svelte';
  import Popover from '$components/ui/Popover.svelte';
  import { resourceViews } from '$lib/resourceLinks';
  import { curationSettingsStore } from '$stores/curationSettings.svelte';

  const links = $derived(resourceViews(curationSettingsStore.settings.resource_links));

  onMount(() => {
    void curationSettingsStore.init();
  });
</script>

<div class="shrink-0" data-testid="resources-menu">
  <Popover label="Resources" align="end" panelClass="w-64 py-1">
    {#snippet trigger({ open, attrs })}
      <button
        type="button"
        class="btn-sm btn-ghost"
        aria-label="Resources"
        data-testid="resources-trigger"
        {...attrs}
      >
        <Icon name="circle-help" size={14} class="lg:hidden" />
        <span class="hidden lg:inline">Resources</span>
        <Icon name={open ? 'chevron-up' : 'chevron-down'} size={14} />
      </button>
    {/snippet}
    <div data-testid="resources-list">
      {#each links as l (l.id)}
        <ResourceLinkItem view={l} class="px-3 py-1.5 text-body hover:bg-surface-2" />
      {/each}
    </div>
  </Popover>
</div>
