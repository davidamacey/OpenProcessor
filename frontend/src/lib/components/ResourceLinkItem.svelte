<script lang="ts">
  /** One served resource entry: an anchor, or a muted row when it has no
   *  usable link. The same markup serves the menu and the dashboards row;
   *  `class` styles the wrapper. */
  import type { ResourceView } from '$lib/resourceLinks';

  let { view, class: cls = '' }: { view: ResourceView; class?: string } = $props();
</script>

<div class="flex items-center gap-2 {cls}" data-key={view.id}>
  {#if view.href}
    <!-- eslint-disable svelte/no-navigation-without-resolve -- same-origin docs paths and served external dashboards, not SvelteKit routes -->
    <a
      href={view.href}
      target="_blank"
      rel="noopener noreferrer"
      data-testid="resource-link"
      title={view.hint || undefined}
      class="text-zinc-200 hover:text-white">{view.label} ↗</a
    >
    <!-- eslint-enable svelte/no-navigation-without-resolve -->
    {#if view.notRunning}
      <span class="text-xs text-amber-400" data-testid="resource-not-running"
        >not running</span
      >
    {/if}
  {:else}
    <span
      class="text-zinc-500"
      data-testid="resource-muted"
      title={view.hint || undefined}>{view.label}: {view.note}</span
    >
  {/if}
</div>
