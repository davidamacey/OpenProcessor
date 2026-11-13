<script lang="ts">
  /**
   * The dashboard's Data quality card: the served accuracy-audit counts and the image-level
   * duplicate run, each linking to its tab. Counts are the server's; a run that never happened says
   * so instead of showing a zero.
   */
  import { resolve } from '$app/paths';
  import { onMount } from 'svelte';
  import Card from '$components/ui/Card.svelte';
  import ErrorState from '$components/ui/ErrorState.svelte';
  import Skeleton from '$components/ui/Skeleton.svelte';
  import { getAuditReport } from '$lib/api_labelConfirmation';
  import { getNearDupStatus } from '$lib/api_nearDup';
  import { formatCount } from '$lib/format/count';
  import { projectHref } from '$lib/projectPaths';
  import type { AuditReport } from '$lib/types_labelConfirmation';
  import type { NearDupLevelStatus } from '$lib/types_nearDup';

  let audit = $state<AuditReport | null>(null);
  let image = $state<NearDupLevelStatus | null>(null);
  let failure = $state<unknown>(null);
  let loading = $state(true);

  async function load(signal?: AbortSignal): Promise<void> {
    loading = true;
    failure = null;
    const [a, n] = await Promise.allSettled([
      getAuditReport(null, signal),
      getNearDupStatus(signal),
    ]);
    if (signal?.aborted) return;
    audit = a.status === 'fulfilled' ? a.value : null;
    image =
      n.status === 'fulfilled'
        ? (n.value.levels.find((l) => l.level === 'image') ?? null)
        : null;
    const bad = [a, n].find((r) => r.status === 'rejected');
    failure = bad && bad.status === 'rejected' ? bad.reason : null;
    loading = false;
  }

  onMount(() => {
    const ctl = new AbortController();
    void load(ctl.signal);
    return () => ctl.abort();
  });

  const run = $derived(image?.current_run ?? null);
</script>

<Card data-testid="data-quality-card">
  <h2 class="mb-3 text-title font-semibold text-fg-1">Data quality</h2>
  {#if failure}
    <ErrorState
      title="Data quality unavailable"
      error={failure}
      onretry={() => void load()}
    />
  {:else if loading}
    <Skeleton lines={2} />
  {:else}
    <div class="grid gap-4 sm:grid-cols-2">
      <div data-testid="dq-audit">
        <p class="text-meta text-fg-3">Accuracy audit</p>
        {#if audit && (audit.audited > 0 || audit.pending > 0)}
          <p class="text-body text-fg-1">
            <span class="tabular-nums">{formatCount(audit.audited)}</span> audited,
            <span class="tabular-nums">{formatCount(audit.pending)}</span> waiting
          </p>
        {:else}
          <p class="text-body text-fg-2">No audit yet</p>
        {/if}
        <a
          class="text-meta text-accent-text hover:underline"
          href={resolve(projectHref('/quality/audit'))}>Open the audit</a
        >
      </div>
      <div data-testid="dq-duplicates">
        <p class="text-meta text-fg-3">Near-duplicates (images)</p>
        {#if run}
          <p class="text-body text-fg-1">
            <span class="tabular-nums">{formatCount(run.dedup_groups)}</span> duplicate
            groups,
            <span class="tabular-nums">{formatCount(run.leak_groups)}</span> leak groups
          </p>
          {#if image && image.pending_count > 0}
            <p class="text-meta text-warn-text">
              {formatCount(image.pending_count)} images added since the run
            </p>
          {/if}
        {:else}
          <p class="text-body text-fg-2">Never run</p>
        {/if}
        <a
          class="text-meta text-accent-text hover:underline"
          href={resolve(projectHref('/quality/near-duplicates'))}>Open near-duplicates</a
        >
      </div>
    </div>
  {/if}
</Card>
