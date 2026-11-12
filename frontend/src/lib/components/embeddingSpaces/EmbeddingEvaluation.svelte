<script lang="ts">
  /**
   * The served evaluation of a space against the primary: one card per run with its verdict, the
   * protocol, the sample and the metrics exactly as stored, and whether the run counts for a
   * promote. Reads only; nothing is computed or ranked here.
   */
  import { metricRows } from '$lib/embeddingSpaces/evaluationMetrics';
  import { formatTimestamp } from '$lib/format/date';
  import type { EmbeddingEvaluationList } from '$lib/types_embeddingSpaces';

  interface Props {
    evaluation: EmbeddingEvaluationList | null;
    error: string | null;
  }

  let { evaluation, error }: Props = $props();
</script>

<div class="flex flex-col gap-2 text-sm" data-testid="space-evaluation">
  {#if error}
    <p class="text-red-300" data-testid="evaluation-error">{error}</p>
  {:else if !evaluation}
    <p class="text-zinc-500">Loading…</p>
  {:else}
    <p class="text-xs text-zinc-400" data-testid="evaluation-summary">
      {#if evaluation.evaluations.length === 0}
        No evaluation of <span class="font-mono">{evaluation.space_key}</span> has been recorded.
      {:else}
        {evaluation.evaluations.length} evaluation{evaluation.evaluations.length === 1
          ? ''
          : 's'} of <span class="font-mono">{evaluation.space_key}</span>.
      {/if}
      {#if evaluation.primary_space_key}
        {evaluation.evaluated_against_primary
          ? 'One is against the current primary'
          : 'None is against the current primary'}
        (<span class="font-mono">{evaluation.primary_space_key}</span>).
      {/if}
    </p>
    {#each evaluation.evaluations as e (e.id)}
      <article
        class="flex flex-col gap-1 rounded border border-zinc-800 p-2"
        data-testid="evaluation-run"
        data-verdict={e.verdict}
      >
        <header class="flex flex-wrap items-baseline gap-2 text-xs">
          <span class="font-mono">{e.candidate_space_key}</span>
          <span class="text-zinc-500">against</span>
          <span class="font-mono">{e.baseline_space_key}</span>
          <span
            class="rounded border border-zinc-600 px-1.5 text-[11px]"
            data-testid="evaluation-verdict">{e.verdict}</span
          >
          {#if e.counts_for_promote}
            <span
              class="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 text-[11px] text-emerald-300"
              data-testid="evaluation-counts">counts for promote</span
            >
          {/if}
          <span class="grow"></span>
          <span class="text-zinc-500"
            >{e.protocol} · {formatTimestamp(e.created_at)} · {e.created_by}</span
          >
        </header>
        <dl class="grid grid-cols-[minmax(0,max-content)_1fr] gap-x-3 text-xs">
          {#each metricRows(e.metrics) as m (m.path)}
            <dt class="font-mono break-all text-zinc-400">{m.path}</dt>
            <dd class="font-mono" data-testid="evaluation-metric">{m.value}</dd>
          {/each}
          {#each metricRows(e.sample_spec, 'sample') as m (m.path)}
            <dt class="font-mono break-all text-zinc-500">{m.path}</dt>
            <dd class="font-mono text-zinc-400">{m.value}</dd>
          {/each}
        </dl>
      </article>
    {/each}
  {/if}
</div>
