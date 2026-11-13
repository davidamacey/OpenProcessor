<script lang="ts">
  /** "Check the draft for activation" and the served report of that check. */
  import ConfigIssueList from '$components/config/ConfigIssueList.svelte';
  import type { ValidationReport } from '$lib/types_config';

  interface Props {
    ed: {
      readonly viewing: unknown;
      readonly activationChecking: boolean;
      readonly activationCheckError: string | null;
      readonly activationReport: ValidationReport | null;
    };
    oncheck: () => void;
  }

  let { ed, oncheck }: Props = $props();
</script>

{#if !ed.viewing}
  <div class="flex flex-col gap-2 border-t border-line pt-2">
    <button
      type="button"
      class="btn btn-sm self-start"
      disabled={ed.activationChecking}
      data-testid="check-activation"
      onclick={oncheck}
      >{ed.activationChecking ? 'Checking…' : 'Check the draft for activation'}</button
    >
    {#if ed.activationCheckError}
      <p class="text-meta text-danger-text">{ed.activationCheckError}</p>
    {/if}
    {#if ed.activationReport}
      {@const r = ed.activationReport}
      <div class="space-y-1" data-testid="activation-check">
        <p class="text-meta text-fg-2">
          As it would be checked on activation: {r.errors.length} error{r.errors
            .length === 1
            ? ''
            : 's'}, {r.warnings.length} warning{r.warnings.length === 1
            ? ''
            : 's'}{r.force_allowed ? '; the server allows overriding' : ''}.
        </p>
        <ConfigIssueList issues={[...r.errors, ...r.warnings]} showField />
      </div>
    {/if}
  </div>
{/if}
