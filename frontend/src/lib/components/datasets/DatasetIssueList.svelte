<script lang="ts">
  /**
   * A dataset's served issues (W10.4 `DatasetIssue`), grouped by the
   * served `severity`, each headed by the served catalog label
   * (`GET /datasets/formats` `issues`) with its own served `message`,
   * `count` and samples. Blocking is the served flag, never inferred.
   */
  import type { DatasetIssue, DatasetIssueCatalogEntry } from '$lib/types_import';

  interface Props {
    issues: DatasetIssue[];
    catalog: DatasetIssueCatalogEntry[];
  }

  let { issues, catalog }: Props = $props();

  const SEVERITIES = [
    { id: 'error', title: 'Errors', tone: 'border-red-900 bg-red-950/30 text-red-200' },
    {
      id: 'warning',
      title: 'Warnings',
      tone: 'border-amber-900 bg-amber-950/30 text-amber-200',
    },
    { id: 'info', title: 'Info', tone: 'border-zinc-800 bg-zinc-900/40 text-zinc-300' },
  ] as const;

  const labelByCode = $derived(new Map(catalog.map((c) => [c.code, c.label])));
  const groups = $derived(
    SEVERITIES.map((s) => ({
      ...s,
      items: issues.filter((i) => i.severity === s.id),
    })).filter((g) => g.items.length > 0),
  );
</script>

{#if issues.length === 0}
  <p class="text-xs text-zinc-500" data-testid="dataset-issues-none">No issues.</p>
{:else}
  <div class="space-y-3" data-testid="dataset-issues">
    {#each groups as g (g.id)}
      <section data-testid="dataset-issues-{g.id}">
        <h4 class="mb-1 text-xs font-semibold tracking-wide text-zinc-400 uppercase">
          {g.title}
        </h4>
        <ul class="space-y-1">
          {#each g.items as issue, idx (`${issue.code}:${idx}`)}
            <li class="rounded border px-3 py-2 text-xs {g.tone}" data-code={issue.code}>
              <div class="flex flex-wrap items-baseline gap-2">
                <span class="font-semibold"
                  >{labelByCode.get(issue.code) ?? issue.code}</span
                >
                <span class="font-mono text-[11px] opacity-70"
                  >{issue.count.toLocaleString()}</span
                >
                {#if issue.blocking}
                  <span class="rounded bg-red-900/60 px-1.5 text-[10px] text-red-100"
                    >blocks the import{issue.bypassable ? ' (can be forced)' : ''}</span
                  >
                {/if}
              </div>
              <p class="mt-0.5">{issue.message}</p>
              {#if issue.samples.length > 0}
                <details class="mt-1">
                  <summary class="cursor-pointer opacity-80">
                    {issue.samples.length} sample{issue.samples.length === 1 ? '' : 's'}
                  </summary>
                  <ul class="mt-1 space-y-0.5 font-mono text-[11px]">
                    {#each issue.samples as s, i (i)}
                      <li>{s.file}{s.line != null ? `:${s.line}` : ''}</li>
                    {/each}
                  </ul>
                </details>
              {/if}
            </li>
          {/each}
        </ul>
      </section>
    {/each}
  </div>
{/if}
