<script lang="ts">
  /**
   * Served validation issues (§7.1 `ValidationIssue`), in served order:
   * severity, message, code, and the field path when asked. Nothing here
   * decides whether an issue blocks; `bypassable` is shown as served.
   */
  import type { ValidationIssue } from '$lib/types_packs';

  interface Props {
    issues: ValidationIssue[];
    /** Show each issue's `field` path (off under a field that owns it). */
    showField?: boolean;
  }

  let { issues, showField = false }: Props = $props();

  const TONE: Record<string, string> = {
    error: 'border-red-500/40 bg-red-500/10 text-red-200',
    warning: 'border-amber-500/40 bg-amber-500/10 text-amber-200',
    info: 'border-zinc-600 bg-zinc-800/60 text-zinc-300',
  };
</script>

{#if issues.length > 0}
  <ul class="space-y-1" data-testid="pack-issues">
    {#each issues as i, idx (idx)}
      <li
        class="rounded border px-2 py-1 text-xs {TONE[i.severity] ?? TONE.info}"
        data-testid="pack-issue"
        data-code={i.code}
        data-severity={i.severity}
      >
        <span class="font-semibold">{i.severity}</span>
        {#if showField && i.field}<code class="ml-1 font-mono opacity-80">{i.field}</code
          >{/if}
        <span class="ml-1">{i.message}</span>
        <code class="ml-1 font-mono text-[10px] opacity-60">{i.code}</code>
        {#if i.bypassable}<span class="ml-1 text-[10px] opacity-80"
            >(can be overridden)</span
          >{/if}
      </li>
    {/each}
  </ul>
{/if}
