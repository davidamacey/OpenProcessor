<script lang="ts">
  /**
   * The Ingest page's sticky Pipeline rail, visible on every tab: how many images are in, how much
   * detection is still pending (the served worklog, polled here so the clustering gate knows on
   * any tab), and the cluster run with its Activate button.
   */
  import ClusterRunPanel from '$components/clusters/ClusterRunPanel.svelte';
  import Card from '$components/ui/Card.svelte';
  import IngestStatusTable from '$lib/components/ingest/IngestStatusTable.svelte';
  import RegionDrainPanel from '$lib/components/ingest/RegionDrainPanel.svelte';
  import type { RegionDrain } from '$lib/types_ingest';

  interface Props {
    refreshToken: number;
    regionConfigured: boolean;
    drainPollIntervalS: number;
    onDrainUpdate: (next: RegionDrain | null, observedAt: number) => void;
  }
  let { refreshToken, regionConfigured, drainPollIntervalS, onDrainUpdate }: Props =
    $props();
</script>

<aside
  class="flex flex-col gap-3 lg:sticky lg:top-4 lg:self-start"
  aria-label="Pipeline"
  data-testid="ingest-pipeline-rail"
>
  <h2 class="text-title font-semibold text-fg-1">Pipeline</h2>
  <Card>
    <IngestStatusTable {refreshToken} />
  </Card>
  {#if regionConfigured}
    <Card data-testid="ingest-rail-detection">
      <RegionDrainPanel pollIntervalS={drainPollIntervalS} onUpdate={onDrainUpdate} />
    </Card>
  {/if}
  <ClusterRunPanel primary={false} />
</aside>
