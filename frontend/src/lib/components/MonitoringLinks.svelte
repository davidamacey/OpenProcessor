<script lang="ts">
  /**
   * Row of external links to the running monitoring / experiment dashboards
   * (MLflow, Grafana, Prometheus, OpenSearch Dashboards).
   *
   * URLs are resolved client-side from the browser's current hostname plus the
   * dashboard's published port, so the links work whether the labeler is opened
   * on localhost or a remote host. Each can be overridden with a PUBLIC_*_URL
   * env var (e.g. PUBLIC_MLFLOW_URL) for non-standard deployments.
   */
  import { onMount } from 'svelte';

  // Highlight MLflow first since train/bake-off runs log there.
  const SERVICES: Array<{ key: string; label: string; port: number; env?: string }> = [
    { key: 'mlflow', label: 'MLflow', port: 5000, env: 'PUBLIC_MLFLOW_URL' },
    { key: 'grafana', label: 'Grafana', port: 4605, env: 'PUBLIC_GRAFANA_URL' },
    { key: 'prometheus', label: 'Prometheus', port: 4604, env: 'PUBLIC_PROMETHEUS_URL' },
    { key: 'opensearch', label: 'OpenSearch', port: 4608, env: 'PUBLIC_OPENSEARCH_DASHBOARDS_URL' },
  ];

  let links = $state<Array<{ label: string; href: string }>>([]);

  onMount(() => {
    const env = import.meta.env as Record<string, string | undefined>;
    const host = window.location.hostname || 'localhost';
    links = SERVICES.map((s) => ({
      label: s.label,
      href: (s.env && env[s.env]) || `http://${host}:${s.port}`,
    }));
  });
</script>

{#if links.length}
  <div class="flex flex-wrap items-center gap-2 text-xs text-zinc-400">
    <span class="text-zinc-500">Dashboards:</span>
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
