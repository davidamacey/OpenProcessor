<script lang="ts">
  /**
   * `/quality`: dataset hygiene in one place. Accuracy audit, near-duplicates
   * and the label history are tabs; each tab is its own route so the URL
   * names it and the browser back button works.
   */
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  import { projectHref } from '$lib/projectPaths';

  let { children } = $props();

  const TABS = [
    { path: '/quality/audit', label: 'Accuracy audit' },
    { path: '/quality/near-duplicates', label: 'Near-duplicates' },
    { path: '/quality/history', label: 'Label history' },
  ] as const;
</script>

<div class="flex h-full flex-col">
  <nav
    class="flex flex-wrap gap-1 border-b border-line px-4 pt-3 sm:px-6"
    aria-label="Quality"
    data-testid="quality-tabs"
  >
    {#each TABS as tab (tab.path)}
      {@const href = resolve(projectHref(tab.path))}
      <a
        {href}
        data-testid="quality-tab-{tab.path.split('/')[2]}"
        aria-current={page.url.pathname === href ? 'page' : undefined}
        class="rounded-t px-3 py-2 text-sm {page.url.pathname === href
          ? 'bg-surface-2 text-fg-1'
          : 'text-fg-2 hover:bg-surface-1 hover:text-fg-1'}">{tab.label}</a
      >
    {/each}
  </nav>
  <div class="min-h-0 grow">{@render children()}</div>
</div>
