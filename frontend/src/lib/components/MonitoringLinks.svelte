<script lang="ts">
  /**
   * Row of external links to the running monitoring / experiment dashboards
   * (MLflow, Grafana, Prometheus, OpenSearch Dashboards).
   *
   * Grafana, Prometheus and OpenSearch Dashboards come only from the served
   * `GET /settings` `monitoring_links`: a null URL, or one that is not an
   * absolute http(s) URL, shows no link (nothing is guessed from the host or
   * a port).
   *
   * MLflow: its link is the served `mlflow_public_url` of `GET /health`
   * (http(s) only); when none is served, the MLflow link is not shown.
   */
  import { onMount } from 'svelte';
  import { externalHref } from '$lib/mlflowLink';
  import { healthStore } from '$stores/health.svelte';
  import { monitoringResourceLinks } from '$lib/resourceLinks';
  import { curationSettingsStore } from '$stores/curationSettings.svelte';

  const mlflowHref = $derived(externalHref(healthStore.health?.mlflow_public_url));
  const links = $derived(
    monitoringResourceLinks(curationSettingsStore.settings.monitoring_links),
  );

  onMount(() => {
    void curationSettingsStore.init();
  });
</script>

{#if mlflowHref || links.length}
  <div class="flex flex-wrap items-center gap-2 text-xs text-zinc-400">
    <span class="text-zinc-500">Dashboards:</span>
    {#if mlflowHref}
      <!-- eslint-disable svelte/no-navigation-without-resolve -- external MLflow server URL, not a SvelteKit route; resolve() only handles in-app routes -->
      <a
        href={mlflowHref}
        target="_blank"
        rel="noopener noreferrer"
        data-testid="mlflow-link"
        title="MLflow server the runs log to (served by the API)"
        class="rounded border border-zinc-700 px-2 py-0.5 text-zinc-300 hover:border-zinc-500 hover:text-white"
      >
        MLflow ↗
      </a>
      <!-- eslint-enable svelte/no-navigation-without-resolve -->
    {/if}
    {#each links as l (l.label)}
      <!-- eslint-disable svelte/no-navigation-without-resolve -- external monitoring dashboard URL (Grafana/Prometheus/OpenSearch), not a SvelteKit route -->
      <a
        href={l.href}
        target="_blank"
        rel="noopener noreferrer"
        data-testid="monitoring-link"
        class="rounded border border-zinc-700 px-2 py-0.5 text-zinc-300 hover:border-zinc-500 hover:text-white"
      >
        {l.label} ↗
      </a>
      <!-- eslint-enable svelte/no-navigation-without-resolve -->
    {/each}
  </div>
{/if}
