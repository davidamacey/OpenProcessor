<script lang="ts">
  /** The training form's inline preflight report. */
  import Icon from '$components/ui/Icon.svelte';
  import type { IconName } from '$components/ui/iconRegistry';
  import { checkSeverityClass } from '$lib/train/trainDisplay';
  import type { PreflightCheck, PreflightReport } from '$lib/types_train';

  interface Props {
    preflight: PreflightReport | null;
    preflighting: boolean;
  }

  let { preflight, preflighting }: Props = $props();

  // Severity is an icon and a word, never colour alone.
  const SEVERITY_ICON: Record<PreflightCheck['severity'], IconName> = {
    ok: 'circle-check',
    warn: 'triangle-alert',
    block: 'circle-x',
  };
  const SEVERITY_WORD: Record<PreflightCheck['severity'], string> = {
    ok: 'OK',
    warn: 'Warning',
    block: 'Blocking',
  };

  // The OK rows are one disclosure: a report of 17 passes buries the one warning.
  const attention = $derived(preflight?.checks.filter((c) => c.severity !== 'ok') ?? []);
  const passed = $derived(preflight?.checks.filter((c) => c.severity === 'ok') ?? []);
</script>

<!-- Preflight panel -->
<section class="rounded-md border border-line bg-surface-1 p-3">
  <header class="mb-2 flex items-center gap-3">
    <h3 class="text-meta text-fg-2 font-semibold">Preflight</h3>
    {#if preflighting}
      <span class="text-caption text-fg-2">checking…</span>
    {:else if preflight}
      <span
        class="rounded-sm border px-1.5 py-0.5 text-caption font-medium capitalize {preflight.blocked
          ? 'border-danger/40 bg-danger/10 text-danger-text'
          : 'border-success/40 bg-success/10 text-success-text'}"
      >
        {preflight.blocked ? 'blocked' : 'ok'}
      </span>
      <span class="text-meta text-fg-2">{preflight.summary ?? ''}</span>
    {/if}
  </header>
  {#if !preflight && !preflighting}
    <p class="text-meta text-fg-3" data-testid="preflight-empty">
      Not checked yet. Run the preflight to check the dataset before a run.
    </p>
  {:else if preflight && preflight.checks.length === 0}
    <p class="text-meta text-fg-3" data-testid="preflight-empty">
      No checks were reported for this dataset.
    </p>
  {/if}
  {#if preflight}
    {#if attention.length > 0}
      <ul class="space-y-1.5" data-testid="preflight-open-rows">
        {#each attention as c (c.name)}
          {@render row(c)}
        {/each}
      </ul>
    {/if}
    {#if passed.length > 0}
      <details class="mt-2" data-testid="preflight-ok-rows">
        <summary class="cursor-pointer text-caption text-fg-3">
          {passed.length} check{passed.length === 1 ? '' : 's'} passed
        </summary>
        <ul class="mt-1.5 space-y-1.5">
          {#each passed as c (c.name)}
            {@render row(c)}
          {/each}
        </ul>
      </details>
    {/if}
  {/if}
</section>

{#snippet row(c: PreflightCheck)}
  <li
    class="flex items-start gap-2 rounded border px-2 py-1.5 text-meta {checkSeverityClass(
      c.severity,
    )}"
  >
    <span class="inline-flex shrink-0 items-center gap-1 text-caption font-medium">
      <Icon name={SEVERITY_ICON[c.severity]} size={14} />
      <span data-testid="preflight-severity">{SEVERITY_WORD[c.severity]}</span>
    </span>
    <span class="grow">
      <span class="font-medium text-fg-1" data-testid="preflight-label">{c.label}</span>:
      {c.message}
      {#if c.detail && Object.keys(c.detail).length > 0}
        <details class="mt-1">
          <summary class="cursor-pointer text-caption text-fg-3">detail</summary>
          <pre
            class="mt-1 text-caption break-all whitespace-pre-wrap text-fg-2">{JSON.stringify(
              c.detail,
              null,
              2,
            )}</pre>
        </details>
      {/if}
    </span>
  </li>
{/snippet}
