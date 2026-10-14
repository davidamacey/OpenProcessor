<script lang="ts">
  /** The training form's inline preflight report. */
  import { checkSeverityClass } from '$lib/train/trainDisplay';
  import type { PreflightReport } from '$lib/types_train';

  interface Props {
    preflight: PreflightReport | null;
    preflighting: boolean;
  }

  let { preflight, preflighting }: Props = $props();
</script>

<!-- Preflight panel -->
<section class="rounded-md border border-zinc-800 bg-zinc-900 p-3">
  <header class="mb-2 flex items-center gap-3">
    <h3 class="text-[11px] uppercase tracking-wide text-zinc-500">Preflight</h3>
    {#if preflighting}
      <span class="text-[11px] text-zinc-400">checking…</span>
    {:else if preflight}
      <span
        class="rounded-sm border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide {preflight.blocked
          ? 'border-red-500/40 bg-red-500/10 text-red-200'
          : 'border-green-500/40 bg-green-500/10 text-green-200'}"
      >
        {preflight.blocked ? 'blocked' : 'ok'}
      </span>
      <span class="text-xs text-zinc-400">{preflight.summary ?? ''}</span>
    {/if}
  </header>
  {#if preflight}
    <ul class="space-y-1.5">
      {#each preflight.checks as c (c.name)}
        <li
          class="flex items-start gap-2 rounded border px-2 py-1.5 text-xs {checkSeverityClass(
            c.severity,
          )}"
        >
          <span class="font-mono text-[10px] uppercase tracking-wide opacity-70">
            {c.severity}
          </span>
          <span class="font-mono text-[10px] text-zinc-400">{c.name}</span>
          <span class="grow">
            {c.message}
            {#if c.detail && Object.keys(c.detail).length > 0}
              <details class="mt-1">
                <summary class="cursor-pointer text-[10px] text-zinc-500">detail</summary>
                <pre
                  class="mt-1 whitespace-pre-wrap break-all font-mono text-[10px] text-zinc-400">{JSON.stringify(
                    c.detail,
                    null,
                    2,
                  )}</pre>
              </details>
            {/if}
          </span>
        </li>
      {/each}
    </ul>
  {/if}
</section>
