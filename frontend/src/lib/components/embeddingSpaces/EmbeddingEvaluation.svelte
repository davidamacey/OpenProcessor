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

<div class="flex flex-col gap-2 text-body" data-testid="space-evaluation">
  {#if error}
    <p class="text-danger-text" data-testid="evaluation-error">{error}</p>
  {:else if !evaluation}
    <p class="text-fg-3">Loading…</p>
  {:else}
    <p class="text-meta text-fg-2" data-testid="evaluation-summary">
      {#if evaluation.evaluations.length === 0}
        No evaluation of <code>{evaluation.space_key}</code> has been recorded.
      {:else}
        {evaluation.evaluations.length} evaluation{evaluation.evaluations.length === 1
          ? ''
          : 's'} of <code>{evaluation.space_key}</code>.
      {/if}
      {#if evaluation.primary_space_key}
        {evaluation.evaluated_against_primary
          ? 'One is against the current primary'
          : 'None is against the current primary'}
        (<code>{evaluation.primary_space_key}</code>).
      {/if}
    </p>
    {#each evaluation.evaluations as e (e.id)}
      <article
        class="flex flex-col gap-1 rounded border border-line p-2"
        data-testid="evaluation-run"
        data-verdict={e.verdict}
      >
        <header class="flex flex-wrap items-baseline gap-2 text-meta">
          <code>{e.candidate_space_key}</code>
          <span class="text-fg-3">against</span>
          <code>{e.baseline_space_key}</code>
          <span
            class="rounded border border-line-strong px-1.5 text-caption"
            data-testid="evaluation-verdict">{e.verdict}</span
          >
          {#if e.counts_for_promote}
            <span
              class="rounded border border-success/40 bg-success/10 px-1.5 text-caption text-success-text"
              data-testid="evaluation-counts">counts for promote</span
            >
          {/if}
          <span class="grow"></span>
          <span class="text-fg-3"
            >{e.protocol} · {formatTimestamp(e.created_at)} · {e.created_by}</span
          >
        </header>
        <dl class="grid grid-cols-[minmax(0,max-content)_1fr] gap-x-3 text-meta">
          {#each metricRows(e.metrics) as m (m.path)}
            <dt class="break-all text-fg-2"><code>{m.path}</code></dt>
            <dd class="tabular-nums" data-testid="evaluation-metric">{m.value}</dd>
          {/each}
          {#each metricRows(e.sample_spec, 'sample') as m (m.path)}
            <dt class="break-all text-fg-3"><code>{m.path}</code></dt>
            <dd class="text-fg-2 tabular-nums">{m.value}</dd>
          {/each}
        </dl>
      </article>
    {/each}
  {/if}
</div>
