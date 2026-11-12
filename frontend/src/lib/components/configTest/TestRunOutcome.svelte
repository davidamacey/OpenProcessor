<script lang="ts" generics="R">
  /**
   * The Run button, the refusal the server served and the result frame every test-on-crop
   * panel (prompt packs, region profiles) shows; the panel renders the result itself.
   */
  import type { Snippet } from 'svelte';
  import ConfigIssueList from '$components/config/ConfigIssueList.svelte';
  import type { ValidationReport } from '$lib/types_config';

  interface Props {
    t: {
      canRun: boolean;
      running: boolean;
      error: string | null;
      errorReport: ValidationReport | null;
      result: R | null;
    };
    onrun: () => void;
    result: Snippet<[R]>;
  }

  let { t, onrun, result }: Props = $props();
</script>

<div>
  <button
    type="button"
    class="btn btn-primary btn-sm"
    disabled={!t.canRun}
    onclick={onrun}
    data-testid="test-run">{t.running ? 'Testing…' : 'Run test'}</button
  >
</div>

{#if t.error}
  <div class="space-y-1" data-testid="test-error">
    <p class="text-danger-text">{t.error}</p>
    {#if t.errorReport}
      <ConfigIssueList
        issues={[...t.errorReport.errors, ...t.errorReport.warnings]}
        showField
      />
    {/if}
  </div>
{/if}

{#if t.result}
  <div class="flex flex-col gap-3 border-t border-line pt-3" data-testid="test-result">
    {@render result(t.result)}
  </div>
{/if}
