<script lang="ts">
  /**
   * Served combine issues (`CombineIssue`): severity, the served message,
   * the project it concerns, the code (humanized only as a title; the id
   * stays visible) and the `detail` object in a disclosure. Nothing here
   * decides whether an issue blocks; `ok` on the preview does.
   */
  import type { CombineIssue } from '$lib/types_combine';

  interface Props {
    issues: CombineIssue[];
    testid?: string;
  }
  let { issues, testid = 'combine-issues' }: Props = $props();

  const TONE: Record<string, string> = {
    error: 'border-red-500/40 bg-red-500/10 text-red-200',
    warning: 'border-amber-500/40 bg-amber-500/10 text-amber-200',
  };
</script>

{#if issues.length > 0}
  <ul class="space-y-1" data-testid={testid}>
    {#each issues as i, idx (idx)}
      {@const severity = i.severity ?? 'error'}
      <li
        class="rounded border px-2 py-1 text-xs {TONE[severity] ?? TONE.error}"
        data-testid="combine-issue"
        data-code={i.code}
        data-severity={severity}
      >
        <span class="font-semibold">{severity}</span>
        {#if i.project}
          <span class="ml-1 rounded bg-zinc-800 px-1 font-mono text-[10px] text-zinc-300"
            >{i.project}</span
          >
        {/if}
        {#if i.message}<span class="ml-1">{i.message}</span>{/if}
        <code class="ml-1 font-mono text-[10px] opacity-60">{i.code}</code>
        {#if i.detail && Object.keys(i.detail).length > 0}
          <details class="mt-1">
            <summary class="cursor-pointer text-[10px] opacity-80">detail</summary>
            <pre
              class="mt-1 max-h-40 overflow-auto rounded bg-zinc-950/60 p-1 text-[10px] text-zinc-300">{JSON.stringify(
                i.detail,
                null,
                2,
              )}</pre>
          </details>
        {/if}
      </li>
    {/each}
  </ul>
{/if}
