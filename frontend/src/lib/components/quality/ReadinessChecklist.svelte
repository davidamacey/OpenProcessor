<script lang="ts">
  /**
   * The served readiness checks as rows: an icon and a word for the severity (never colour
   * alone), the served label and message, and one link to the items or page that need work.
   * Everything shown is what `GET /quality/readiness` served; the link is the served
   * `section` plus `query`.
   */
  import { resolve } from '$app/paths';
  import Icon from '$components/ui/Icon.svelte';
  import type { IconName } from '$components/ui/iconRegistry';
  import { projectHref, type ProjectSubPath } from '$lib/projectPaths';
  import type { ReadinessCheck, ReadinessSeverity } from '$lib/types_quality';

  let { checks }: { checks: ReadinessCheck[] } = $props();

  const SEVERITY: Record<
    ReadinessSeverity,
    { icon: IconName; word: string; tone: string }
  > = {
    block: { icon: 'circle-x', word: 'Blocking', tone: 'text-danger-text' },
    warn: { icon: 'triangle-alert', word: 'Warning', tone: 'text-warn-text' },
    ok: { icon: 'circle-check', word: 'OK', tone: 'text-success-text' },
  };

  // Where a served link section leads, in words.
  const OPEN: Record<string, string> = {
    workspace: 'Open in Workspace',
    export: 'Open Export',
    'quality/audit': 'Open Accuracy audit',
    classes: 'Open Classes',
  };

  const hrefOf = (c: ReadinessCheck): string | null =>
    c.link
      ? projectHref(
          `/${c.link.section}${c.link.query ? `?${c.link.query}` : ''}` as ProjectSubPath,
        )
      : null;
</script>

<ul class="divide-y divide-line rounded-card border border-line bg-surface-1">
  {#each checks as check (check.id)}
    {@const sev = SEVERITY[check.severity]}
    {@const href = hrefOf(check)}
    <li
      class="flex flex-wrap items-start gap-x-4 gap-y-1 px-4 py-3"
      data-testid="readiness-row"
      data-check={check.id}
      data-severity={check.severity}
    >
      <span
        class="inline-flex w-24 shrink-0 items-center gap-1.5 text-control font-medium {sev.tone}"
        data-testid="readiness-severity-cell"
      >
        <Icon name={sev.icon} size={16} />
        <span data-testid="readiness-severity">{sev.word}</span>
      </span>
      <div class="min-w-0 flex-1">
        <p class="text-body font-medium text-fg-1">{check.label}</p>
        <p class="text-body text-fg-2">{check.message}</p>
        {#if check.classes.length > 0}
          <p class="mt-0.5 text-meta text-fg-3 tabular-nums">
            {check.classes
              .map((c) => `${c.class_name} ${c.count.toLocaleString()}`)
              .join(' · ')}
          </p>
        {/if}
      </div>
      {#if href}
        <a class="btn btn-sm shrink-0" href={resolve(href as '/')}
          >{OPEN[check.link?.section ?? ''] ?? 'Open'}</a
        >
      {/if}
    </li>
  {/each}
</ul>
