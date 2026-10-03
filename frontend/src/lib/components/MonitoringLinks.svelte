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
   * MLflow (T1, visual audit 2026-09-24): its link comes from the served
   * runs' `mlflow_run_url` origin (see `mlflowBaseUrl`), or an explicit
   * `PUBLIC_MLFLOW_URL`; with neither, the MLflow link is not shown.
   */
  import { onMount } from 'svelte';
  import { mlflowBaseUrl } from '$lib/mlflowLink';
  import { monitoringResourceLinks } from '$lib/resourceLinks';
  import { curationSettingsStore } from '$stores/curationSettings.svelte';

  interface Props {
    /** Served `mlflow_run_url`s of the runs on this page, if any. */
    mlflowRunUrls?: Array<string | null | undefined>;
  }

  let { mlflowRunUrls = [] }: Props = $props();

  let mlflowEnv = $state<string | null>(null);

  const mlflowHref = $derived(mlflowBaseUrl(mlflowRunUrls, mlflowEnv));
  const links = $derived(
    monitoringResourceLinks(curationSettingsStore.settings.monitoring_links),
  );

  onMount(() => {
    const env = import.meta.env as Record<string, string | undefined>;
    mlflowEnv = env.PUBLIC_MLFLOW_URL ?? null;
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
        title="MLflow server the runs below log to (from their served run URLs)"
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
