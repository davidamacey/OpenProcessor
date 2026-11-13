<script lang="ts">
  /**
   * Data quality › Ready to train: the dataset-level checklist before a training run. Every
   * row, severity, count and link is served by `GET /quality/readiness`; the page only renders
   * it, and the verdict sentence counts the served severities.
   */
  import { onMount } from 'svelte';
  import Button from '$components/ui/Button.svelte';
  import Chip from '$components/ui/Chip.svelte';
  import ErrorState from '$components/ui/ErrorState.svelte';
  import Section from '$components/ui/Section.svelte';
  import CoverageTable from '$lib/components/quality/CoverageTable.svelte';
  import ReadinessChecklist from '$lib/components/quality/ReadinessChecklist.svelte';
  import { getReadiness } from '$lib/api_quality';
  import { formatTime } from '$lib/format/time';
  import { plural } from '$lib/format/plural';
  import type { DatasetReadiness } from '$lib/types_quality';

  let readiness = $state<DatasetReadiness | null>(null);
  let error = $state<unknown>(null);
  let loading = $state(true);

  async function load(signal?: AbortSignal): Promise<void> {
    loading = true;
    try {
      readiness = await getReadiness(signal);
      error = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      error = e;
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    const ctl = new AbortController();
    void load(ctl.signal);
    return () => ctl.abort();
  });

  const counts = $derived({
    block: readiness?.checks.filter((c) => c.severity === 'block').length ?? 0,
    warn: readiness?.checks.filter((c) => c.severity === 'warn').length ?? 0,
  });
  const verdict = $derived.by(() => {
    if (!readiness) return null;
    const warnings = plural(counts.warn, 'warning');
    if (readiness.verdict === 'blocked')
      return {
        tone: 'danger' as const,
        text: `Not ready: ${counts.block} blocking, ${warnings}`,
      };
    if (readiness.verdict === 'warn')
      return { tone: 'warn' as const, text: `Ready, with ${warnings}` };
    return { tone: 'success' as const, text: 'Ready to train' };
  });
  const checkedAt = $derived(
    readiness ? formatTime(Date.parse(readiness.computed_at), '') : '',
  );
</script>

<div class="mt-4 flex flex-col gap-4">
  <p class="max-w-3xl text-body text-fg-2">
    Check the dataset before you train: enough checked examples per class, a fair test
    set, and machine labels you can trust. Each line links to the items that need work.
  </p>

  {#if error}
    <ErrorState {error} onretry={() => void load()} />
  {:else if !readiness}
    <p class="text-body text-fg-3">Loading…</p>
  {:else}
    <div class="flex flex-wrap items-center gap-3">
      {#if verdict}
        <Chip tone={verdict.tone} title="The overall result of the checks below">
          <span data-testid="readiness-verdict">{verdict.text}</span>
        </Chip>
      {/if}
      <span class="grow"></span>
      {#if checkedAt}<span class="text-meta text-fg-3">checked {checkedAt}</span>{/if}
      <Button icon="refresh-cw" disabled={loading} onclick={() => void load()}
        >Refresh</Button
      >
    </div>

    <ReadinessChecklist checks={readiness.checks} />

    <Section title="Coverage per class">
      <CoverageTable coverage={readiness.coverage} />
    </Section>
  {/if}
</div>
