<script lang="ts">
  /**
   * The server's dry run, shown as served: how many members the target has,
   * how many the VLM will see, the estimated VLM calls, the strata of a
   * stratified sample, what was skipped, and the endpoint and pack the run
   * pinned. Nothing here is computed; a refusal is shown verbatim.
   */
  import { formatCount } from '$lib/format/count';
  import type { AssistRunState } from '$lib/types_assist';

  interface Props {
    plan: AssistRunState | null;
    planning: boolean;
    error: string | null;
    errorCode: string | null;
  }

  let { plan, planning, error, errorCode }: Props = $props();

  const SKIPPED_LABELS: Record<string, string> = {
    holdout: 'test holdout',
    excluded: 'ignored',
    class_locked: 'class locked (answers kept as suggestions)',
    box_locked: 'locked boxes',
    box_not_found: 'boxes no longer there',
    not_found: 'not found',
  };

  const resolved = $derived(plan?.resolved ?? null);
  const skipped = $derived(
    Object.entries(resolved?.skipped ?? {}).filter(([, n]) => (n ?? 0) > 0),
  );
  const strata = $derived(resolved?.strata ?? []);
  const named = (o: Record<string, unknown> | null | undefined) =>
    o && typeof o.name === 'string'
      ? `${o.name}${typeof o.revision === 'number' ? ` r${o.revision}` : ''}`
      : null;
</script>

<div
  class="rounded-md border border-zinc-800 bg-zinc-900/60 p-3 text-xs"
  data-testid="assist-plan"
  aria-live="polite"
>
  {#if error}
    <p class="text-red-300" data-testid="assist-plan-error" data-code={errorCode}>
      {error}
    </p>
  {:else if !resolved}
    <p class="text-zinc-500">{planning ? 'Asking the server for a dry run…' : '—'}</p>
  {:else}
    <p class="text-zinc-200" class:opacity-60={planning}>
      <strong data-testid="assist-plan-sampled">{formatCount(resolved.sampled)}</strong>
      of {formatCount(resolved.scope_count)} members go to the VLM, about
      <strong data-testid="assist-plan-calls"
        >{formatCount(resolved.estimated_vlm_calls)}</strong
      >
      VLM calls. Nothing is written until you start the run.
    </p>
    {#if named(resolved.vlm) || named(resolved.prompt_pack)}
      <p class="mt-1 text-zinc-400" data-testid="assist-plan-pinned">
        VLM {named(resolved.vlm) ?? '—'}{typeof resolved.vlm?.model === 'string'
          ? ` (${resolved.vlm.model})`
          : ''} · prompt pack {named(resolved.prompt_pack) ?? '—'}
      </p>
    {/if}
    {#if strata.length > 1}
      <ul
        class="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-zinc-400"
        data-testid="assist-plan-strata"
      >
        {#each strata as s (s.key)}
          <li>
            <span class="text-zinc-300">{s.key === '' ? 'no sub-cluster' : s.key}</span>
            {s.sampled}/{s.size}
          </li>
        {/each}
      </ul>
    {/if}
    {#if skipped.length > 0}
      <p class="mt-2 text-zinc-500" data-testid="assist-plan-skipped">
        Left out:
        {#each skipped as [k, n], i (k)}{i > 0 ? ', ' : ''}{formatCount(n)}
          {SKIPPED_LABELS[k] ?? k}{/each}
      </p>
    {/if}
  {/if}
</div>
