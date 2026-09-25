<script lang="ts">
  /**
   * Row of external links to the running monitoring / experiment dashboards
   * (MLflow, Grafana, Prometheus, OpenSearch Dashboards).
   *
   * URLs are resolved client-side from the browser's current hostname plus the
   * dashboard's published port, so the links work whether the labeler is opened
   * on localhost or a remote host. Each can be overridden with a PUBLIC_*_URL
   * env var (e.g. PUBLIC_MLFLOW_URL) for non-standard deployments.
   *
   * MLflow is the exception (T1, visual audit 2026-09-24): its link comes
   * from the served runs' `mlflow_run_url` origin (see `mlflowBaseUrl`),
   * never a hardcoded port. With no served URL and no PUBLIC_MLFLOW_URL,
   * the MLflow link is not shown.
   */
  import { onMount } from 'svelte';
  import { mlflowBaseUrl } from '$lib/mlflowLink';

  interface Props {
    /** Served `mlflow_run_url`s of the runs on this page, if any. */
    mlflowRunUrls?: Array<string | null | undefined>;
  }

  let { mlflowRunUrls = [] }: Props = $props();

  const SERVICES: Array<{ key: string; label: string; port: number; env?: string }> = [
    { key: 'grafana', label: 'Grafana', port: 4605, env: 'PUBLIC_GRAFANA_URL' },
    { key: 'prometheus', label: 'Prometheus', port: 4604, env: 'PUBLIC_PROMETHEUS_URL' },
    {
      key: 'opensearch',
      label: 'OpenSearch',
      port: 4608,
      env: 'PUBLIC_OPENSEARCH_DASHBOARDS_URL',
    },
  ];

  let links = $state<Array<{ label: string; href: string }>>([]);
  let mlflowEnv = $state<string | null>(null);

  const mlflowHref = $derived(mlflowBaseUrl(mlflowRunUrls, mlflowEnv));

  onMount(() => {
    const env = import.meta.env as Record<string, string | undefined>;
    const host = window.location.hostname || 'localhost';
    mlflowEnv = env.PUBLIC_MLFLOW_URL ?? null;
    links = SERVICES.map((s) => ({
      label: s.label,
      href: (s.env && env[s.env]) || `http://${host}:${s.port}`,
    }));
  });
</script>

{#if links.length}
  <div class="flex flex-wrap items-center gap-2 text-xs text-zinc-400">
    <span class="text-zinc-500">Dashboards:</span>
    {#if mlflowHref}
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
    {/if}
    {#each links as l (l.label)}
      <a
        href={l.href}
        target="_blank"
        rel="noopener noreferrer"
        class="rounded border border-zinc-700 px-2 py-0.5 text-zinc-300 hover:border-zinc-500 hover:text-white"
      >
        {l.label} ↗
      </a>
    {/each}
  </div>
{/if}
