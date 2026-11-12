<script lang="ts">
  /**
   * Served validation issues (§7.1 `ValidationIssue`), in served order:
   * severity, message, code, and the field path when asked. Nothing here
   * decides whether an issue blocks; `bypassable` is shown as served.
   */
  import type { ValidationIssue } from '$lib/types_config';

  interface Props {
    issues: ValidationIssue[];
    /** Show each issue's `field` path (off under a field that owns it). */
    showField?: boolean;
  }

  let { issues, showField = false }: Props = $props();

  const TONE: Record<string, string> = {
    error: 'border-danger/40 bg-danger/10 text-danger-text',
    warning: 'border-warn/40 bg-warn/10 text-warn-text',
    info: 'border-line-strong bg-surface-2/60 text-fg-2',
  };
</script>

{#if issues.length > 0}
  <ul class="space-y-1" data-testid="config-issues">
    {#each issues as i (i.id)}
      <li
        class="rounded border px-2 py-1 text-meta {TONE[i.severity] ?? TONE.info}"
        data-testid="config-issue"
        data-code={i.code}
        data-severity={i.severity}
      >
        <span class="font-semibold">{i.severity}</span>
        {#if showField && i.field}<code class="ml-1 font-mono opacity-80">{i.field}</code
          >{/if}
        <span class="ml-1">{i.message}</span>
        <code class="ml-1 font-mono text-caption opacity-60">{i.code}</code>
        {#if i.bypassable}<span class="ml-1 text-caption opacity-80"
            >(can be overridden)</span
          >{/if}
      </li>
    {/each}
  </ul>
{/if}
