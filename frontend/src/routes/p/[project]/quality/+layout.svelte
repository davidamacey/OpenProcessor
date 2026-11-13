<script lang="ts">
  /**
   * Data quality: the final dataset-level review before training. Ready to train (the default),
   * the accuracy audit and the label history are URL-bound tabs (each its own route, so reload
   * and Back work).
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
    { id: 'ready', path: '/quality/ready', label: 'Ready to train' },
    { id: 'audit', path: '/quality/audit', label: 'Accuracy audit' },
    { id: 'history', path: '/quality/history', label: 'Label history' },
  ] as const;
  const current = $derived(
    TABS.find((t) => page.url.pathname.endsWith(t.path))?.id ?? TABS[0].id,
  );
</script>

<PageShell>
  <PageHeader
    title="Data quality"
    description="Check the dataset before you train: how complete it is, how accurate the machine labels are, and what changed."
  />
  <Tabs
    label="Data quality"
    value={current}
    items={TABS.map((t) => ({ id: t.id, label: t.label, href: projectHref(t.path) }))}
  />
  {@render children()}
</PageShell>
