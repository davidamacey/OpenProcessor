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
    error: 'border-danger/40 bg-danger/10 text-danger-text',
    warning: 'border-warn/40 bg-warn/10 text-warn-text',
  };
</script>

{#if issues.length > 0}
  <ul class="space-y-1" data-testid={testid}>
    {#each issues as i (i.id)}
      {@const severity = i.severity ?? 'error'}
      <li
        class="rounded border px-2 py-1 text-meta {TONE[severity] ?? TONE.error}"
        data-testid="combine-issue"
        data-code={i.code}
        data-severity={severity}
      >
        <span class="font-semibold">{severity}</span>
        {#if i.project}
          <code class="ml-1 rounded bg-surface-2 px-1 text-caption text-fg-2"
            >{i.project}</code
          >
        {/if}
        <span class="ml-1">{i.message}</span>
        <code class="ml-1 text-caption opacity-60">{i.code}</code>
        {#if i.detail && Object.keys(i.detail).length > 0}
          <details class="mt-1">
            <summary class="cursor-pointer text-caption opacity-80">detail</summary>
            <pre
              class="mt-1 max-h-40 overflow-auto rounded bg-surface-0/60 p-1 text-caption text-fg-2">{JSON.stringify(
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
