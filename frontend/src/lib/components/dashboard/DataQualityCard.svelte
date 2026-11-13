<script lang="ts">
  /**
   * The dashboard's Data quality card: the served "ready to train" verdict, the checks that
   * still need work (served label and message), and the way into Data quality and into the
   * Workspace Duplicates view. Everything is what `GET /quality/readiness` served.
   */
  import { resolve } from '$app/paths';
  import { onMount } from 'svelte';
  import Card from '$components/ui/Card.svelte';
  import Chip from '$components/ui/Chip.svelte';
  import ErrorState from '$components/ui/ErrorState.svelte';
  import Skeleton from '$components/ui/Skeleton.svelte';
  import { getReadiness } from '$lib/api_quality';
  import { plural } from '$lib/format/plural';
  import { projectHref } from '$lib/projectPaths';
  import type { DatasetReadiness, ReadinessVerdict } from '$lib/types_quality';

  const VERDICT: Record<
    ReadinessVerdict,
    { tone: 'success' | 'warn' | 'danger'; text: string }
  > = {
    ready: { tone: 'success', text: 'Ready to train' },
    warn: { tone: 'warn', text: 'Ready, with warnings' },
    blocked: { tone: 'danger', text: 'Not ready' },
  };

  let readiness = $state<DatasetReadiness | null>(null);
  let failure = $state<unknown>(null);
  let loading = $state(true);

  async function load(signal?: AbortSignal): Promise<void> {
    loading = true;
    failure = null;
    try {
      readiness = await getReadiness(signal);
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      failure = e;
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    const ctl = new AbortController();
    void load(ctl.signal);
    return () => ctl.abort();
  });

  const open = $derived(readiness?.checks.filter((c) => c.severity !== 'ok') ?? []);
</script>

<Card data-testid="data-quality-card">
  <div class="mb-3 flex flex-wrap items-center gap-2">
    <h2 class="text-title font-semibold text-fg-1">Data quality</h2>
    {#if readiness}
      <Chip tone={VERDICT[readiness.verdict].tone}
        ><span data-testid="dq-verdict">{VERDICT[readiness.verdict].text}</span></Chip
      >
    {/if}
  </div>
  {#if failure}
    <ErrorState
      title="Data quality unavailable"
      error={failure}
      onretry={() => void load()}
    />
  {:else if loading}
    <Skeleton lines={2} />
  {:else if readiness}
    {#if open.length > 0}
      <ul class="mb-3 flex flex-col gap-1.5" data-testid="dq-open-checks">
        {#each open.slice(0, 4) as check (check.id)}
          <li class="text-body text-fg-2">
            <span class="font-medium text-fg-1">{check.label}</span>: {check.message}
          </li>
        {/each}
      </ul>
      {#if open.length > 4}
        <p class="mb-3 text-meta text-fg-3">
          and {plural(open.length - 4, 'more check')}
        </p>
      {/if}
    {:else}
      <p class="mb-3 text-body text-fg-2">Every check passed.</p>
    {/if}
  {/if}
  <div class="flex flex-wrap gap-x-4 gap-y-1 text-meta">
    <a
      class="text-accent-text hover:underline"
      data-testid="dq-open-ready"
      href={resolve(projectHref('/quality/ready'))}>Open Data quality</a
    >
    <a
      class="text-accent-text hover:underline"
      data-testid="dq-open-duplicates"
      href={resolve(projectHref('/workspace?view=gallery&near_dup=collapsed'))}
      >Open duplicates in the Gallery</a
    >
  </div>
</Card>
