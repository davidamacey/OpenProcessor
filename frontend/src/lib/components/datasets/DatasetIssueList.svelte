<script lang="ts">
  /**
   * A dataset's served issues (`DatasetIssue`), grouped by the
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
    {
      id: 'error',
      title: 'Errors',
      tone: 'border-danger-line bg-danger-soft text-danger-text',
    },
    {
      id: 'warning',
      title: 'Warnings',
      tone: 'border-warn-line bg-warn-soft text-warn-text',
    },
    { id: 'info', title: 'Info', tone: 'border-line bg-surface-1/40 text-fg-2' },
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
  <p class="text-meta text-fg-3" data-testid="dataset-issues-none">No issues.</p>
{:else}
  <div class="space-y-3" data-testid="dataset-issues">
    {#each groups as g (g.id)}
      <section data-testid="dataset-issues-{g.id}">
        <h4 class="mb-1 text-meta font-semibold text-fg-2">
          {g.title}
        </h4>
        <ul class="space-y-1">
          {#each g.items as issue (issue.id)}
            <li
              class="rounded border px-3 py-2 text-meta {g.tone}"
              data-code={issue.code}
            >
              <div class="flex flex-wrap items-baseline gap-2">
                <span class="font-semibold"
                  >{labelByCode.get(issue.code) ?? issue.code}</span
                >
                <span class="text-caption opacity-70 tabular-nums"
                  >{issue.count.toLocaleString()}</span
                >
                {#if issue.blocking}
                  <span
                    class="rounded bg-danger-soft px-1.5 text-caption text-danger-text"
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
                  <ul class="mt-1 space-y-0.5 text-caption">
                    <code
                      >{#each issue.samples as s, i (i)}
                        <li>{s.file}{s.line != null ? `:${s.line}` : ''}</li>
                      {/each}</code
                    >
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
