<script lang="ts">
  /*
   * Pipeline dashboard route: the app's single home page. Top to bottom: the Next steps
   * stepper (served, in loop order), four headline numbers, Labeled by source, the clustering card beside the
   * Data quality card, the class balance, then "Pipeline details" and "Recent labels" as
   * collapsed disclosures. Refresh is the header action; the YOLO export lives on Model >
   * Export and VLM activation in Next steps and Settings.
   */
  import { apiBase } from '$lib/api';
  import Button from '$components/ui/Button.svelte';
  import PageHeader from '$components/ui/PageHeader.svelte';
  import PageShell from '$components/ui/PageShell.svelte';
  import AdoptDetectorDialog from '$components/detector/AdoptDetectorDialog.svelte';
  import NextStepsCard from '$lib/components/NextStepsCard.svelte';
  import { nextStepsStore } from '$stores/nextSteps.svelte';
  import { projectsStore } from '$stores/projects.svelte';
  import { healthStore } from '$stores/health.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import ClusterRunPanel from '$components/clusters/ClusterRunPanel.svelte';
  import DashboardKpis from '$components/dashboard/DashboardKpis.svelte';
  import LabeledBySourceCard from '$components/dashboard/LabeledBySourceCard.svelte';
  import DataQualityCard from '$components/dashboard/DataQualityCard.svelte';
  import DatasetStats from '$components/DatasetStats.svelte';
  import DetectionsSummaryPanel from '$lib/components/detector/DetectionsSummaryPanel.svelte';
  import ClassBalanceChart from '$components/ClassBalanceChart.svelte';
  import { createDatasetStatsFeed } from '$lib/datasetStatsFeed.svelte';
  import { getCrops, getThumbUrl } from '$lib/api_cropRead';
  import { getTestHoldoutStats } from '$lib/api_export';
  import { getStats } from '$lib/api_stats';
  import type { Crop } from '$lib/types';
  import type { TestHoldoutStats } from '$lib/types_export';
  import type { StatsSummary } from '$lib/types_health';

  $effect(() => {
    keyboardStore.setScope('dashboard');
  });

  $effect(() => {
    void nextStepsStore.init();
  });

  // One live stats stream feeds the headline numbers and the pipeline details.
  const feed = createDatasetStatsFeed();

  let adoptOpen = $state(false);

  let legacyStats = $state<StatsSummary | null>(null);
  // Served per-class test holdout, so the balance chart shows trainable counts.
  // null = unavailable; the chart says so rather than guessing.
  let holdoutStats = $state<TestHoldoutStats | null>(null);
  let recent = $state<Crop[]>([]);
  let loading = $state<boolean>(false);

  async function refresh(): Promise<void> {
    loading = true;
    try {
      const [s, c, h] = await Promise.allSettled([
        getStats(),
        getCrops({
          label_validated: true,
          sort: 'updated_at:desc',
          limit: 20,
        }),
        getTestHoldoutStats(),
      ]);
      legacyStats = s.status === 'fulfilled' ? s.value : null;
      holdoutStats = h.status === 'fulfilled' ? h.value : null;
      recent = c.status === 'fulfilled' ? c.value.items : [];
    } finally {
      loading = false;
    }
    void nextStepsStore.refresh();
  }

  $effect(() => {
    void refresh();
    const id = setInterval(() => void refresh(), 30_000);
    return () => clearInterval(id);
  });
</script>

<PageShell width="wide" class="space-y-6">
  <PageHeader
    title="Dashboard"
    description="Where the dataset stands and what to do next."
  >
    {#snippet actions()}
      <Button icon="refresh-cw" onclick={() => void refresh()} disabled={loading}
        >{loading ? 'Refreshing…' : 'Refresh'}</Button
      >
    {/snippet}
  </PageHeader>

  {#if !healthStore.ok}
    <div
      class="rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-body text-danger-text"
    >
      <strong>API unavailable.</strong> Check that the OpenProcessor API is reachable from
      <code>{apiBase || (typeof window !== 'undefined' ? window.location.host : '')}</code
      >.
    </div>
  {/if}

  {#if nextStepsStore.steps.length > 0 && projectsStore.current}
    <NextStepsCard
      steps={nextStepsStore.steps}
      next={nextStepsStore.next}
      slug={projectsStore.current.slug}
      onaction={(step) => {
        if (step.action === 'adopt_detector_classes') adoptOpen = true;
      }}
    />
  {/if}

  <DashboardKpis stats={feed.stats} />
  <LabeledBySourceCard stats={feed.stats} />

  <div class="grid gap-6 lg:grid-cols-2">
    <ClusterRunPanel primary={false} />
    <DataQualityCard />
  </div>

  <!-- Class balance: ClassBalanceChart. -->
  <section class="surface p-4">
    {#if !legacyStats}
      <p class="text-body text-fg-3">Loading...</p>
    {:else}
      <ClassBalanceChart
        rows={legacyStats.per_class}
        holdout={holdoutStats}
        thresholds={legacyStats.thresholds ?? null}
      />
    {/if}
  </section>

  <details class="surface p-4" data-testid="pipeline-details">
    <summary class="cursor-pointer text-title font-semibold text-fg-1"
      >Pipeline details</summary
    >
    <div class="mt-4 space-y-6">
      <DatasetStats {feed} showLabeled={false} />
      <DetectionsSummaryPanel />
    </div>
  </details>

  <details class="surface p-4" data-testid="recent-labels">
    <summary class="cursor-pointer text-title font-semibold text-fg-1"
      >Recent labels</summary
    >
    {#if recent.length === 0}
      <p class="mt-3 text-body text-fg-3">No recent labels.</p>
    {:else}
      <ul class="mt-3 grid grid-cols-3 gap-3 sm:grid-cols-5 lg:grid-cols-10">
        {#each recent as crop (crop.id)}
          <li class="rounded-md border border-line bg-surface-1">
            <img
              src={getThumbUrl(crop.id)}
              alt="crop"
              loading="lazy"
              class="aspect-square w-full rounded-t-md object-contain"
            />
            <div
              class="truncate px-1.5 py-1 text-caption text-fg-2"
              title={crop.class_name ?? ''}
            >
              {crop.class_name ?? '—'}
            </div>
          </li>
        {/each}
      </ul>
    {/if}
  </details>
</PageShell>

{#if adoptOpen}
  <AdoptDetectorDialog
    onclose={() => (adoptOpen = false)}
    ondone={() => void nextStepsStore.refresh()}
  />
{/if}
