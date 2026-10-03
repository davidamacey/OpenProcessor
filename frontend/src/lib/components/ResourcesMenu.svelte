<script lang="ts">
  /**
   * Top-bar "Resources" menu: bundled docs, the API's interactive docs
   * (same-origin proxies) and every served dashboard link. Plain external
   * anchors, so no resolve().
   */
  import { onMount } from 'svelte';
  import { externalHref } from '$lib/mlflowLink';
  import { healthStore } from '$stores/health.svelte';
  import { resourceLinks } from '$lib/resourceLinks';
  import { curationSettingsStore } from '$stores/curationSettings.svelte';

  let open = $state(false);
  let root = $state<HTMLDivElement | null>(null);
  let trigger = $state<HTMLButtonElement | null>(null);

  const links = $derived(
    resourceLinks(
      curationSettingsStore.settings.monitoring_links,
      externalHref(healthStore.health?.mlflow_public_url),
    ),
  );

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
    <!-- eslint-disable svelte/no-navigation-without-resolve -- same-origin proxy paths and served external dashboards, not SvelteKit routes -->
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <div
      class="absolute right-0 top-full z-50 mt-1 w-64 rounded border border-zinc-700 bg-zinc-900 py-1 shadow-xl"
      role="group"
      aria-label="Resources"
      data-testid="resources-list"
      tabindex="-1"
      onkeydown={onKeydown}
    >
      {#each links as l (l.key)}
        <a
          href={l.href}
          target="_blank"
          rel="noopener noreferrer"
          data-testid="resource-link"
          data-key={l.key}
          class="block px-3 py-1.5 text-sm text-zinc-200 hover:bg-zinc-800 hover:text-white"
          >{l.label} ↗</a
        >
      {/each}
    </div>
    <!-- eslint-enable svelte/no-navigation-without-resolve -->
  {/if}
</div>
