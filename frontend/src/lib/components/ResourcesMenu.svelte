<script lang="ts">
  /**
   * Top-bar "Resources" menu: the bundled docs (client-owned) then exactly
   * the served `resource_links`. Plain external anchors, so no resolve().
   */
  import { onMount } from 'svelte';
  import ResourceLinkItem from '$components/ResourceLinkItem.svelte';
  import { resourceViews } from '$lib/resourceLinks';
  import { curationSettingsStore } from '$stores/curationSettings.svelte';

  let open = $state(false);
  let root = $state<HTMLDivElement | null>(null);
  let trigger = $state<HTMLButtonElement | null>(null);

  const links = $derived(resourceViews(curationSettingsStore.settings.resource_links));

  onMount(() => {
    void curationSettingsStore.init();
  });

  function onWindowPointer(e: PointerEvent): void {
    if (open && root && !root.contains(e.target as Node)) open = false;
  }

  function onKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape' && open) {
      e.stopPropagation();
      open = false;
      // The trigger stays mounted, so focus can move before the list unmounts.
      trigger?.focus();
    }
  }
</script>

<svelte:window onpointerdown={onWindowPointer} />

<div class="relative shrink-0" bind:this={root} data-testid="resources-menu">
  <button
    type="button"
    bind:this={trigger}
    class="rounded px-1 text-sm text-zinc-300 hover:text-white"
    aria-haspopup="true"
    aria-expanded={open}
    data-testid="resources-trigger"
    onclick={() => (open = !open)}
    onkeydown={onKeydown}
  >
    Resources <span class="text-zinc-500" aria-hidden="true">▾</span>
  </button>
  {#if open}
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <div
      class="absolute right-0 top-full z-50 mt-1 w-64 rounded border border-zinc-700 bg-zinc-900 py-1 shadow-xl"
      role="group"
      aria-label="Resources"
      data-testid="resources-list"
      tabindex="-1"
      onkeydown={onKeydown}
    >
      {#each links as l (l.id)}
        <ResourceLinkItem view={l} class="px-3 py-1.5 text-sm hover:bg-zinc-800" />
      {/each}
    </div>
  {/if}
</div>
