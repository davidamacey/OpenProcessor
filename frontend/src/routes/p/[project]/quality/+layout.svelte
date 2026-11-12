<script lang="ts">
  /**
   * Data quality: dataset hygiene in one place. The accuracy audit, near-duplicates and the label
   * history are URL-bound tabs (each its own route, so reload and Back work).
   *
   * Docs read: SvelteKit routing, nested layouts (svelte.dev/docs/kit/routing#layout).
   */
  import { page } from '$app/state';
  import PageHeader from '$components/ui/PageHeader.svelte';
  import PageShell from '$components/ui/PageShell.svelte';
  import Tabs from '$components/ui/Tabs.svelte';
  import { projectHref } from '$lib/projectPaths';

  let { children } = $props();

  const TABS = [
    { id: 'audit', path: '/quality/audit', label: 'Accuracy audit' },
    { id: 'near-duplicates', path: '/quality/near-duplicates', label: 'Near-duplicates' },
    { id: 'history', path: '/quality/history', label: 'Label history' },
  ] as const;
  const current = $derived(
    TABS.find((t) => page.url.pathname.endsWith(t.path))?.id ?? TABS[0].id,
  );
</script>

<PageShell>
  <PageHeader
    title="Data quality"
    description="How accurate the labels are, which images repeat, and what changed."
  />
  <Tabs
    label="Data quality"
    value={current}
    items={TABS.map((t) => ({ id: t.id, label: t.label, href: projectHref(t.path) }))}
  />
  {@render children()}
</PageShell>
