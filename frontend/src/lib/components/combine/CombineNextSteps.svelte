<script lang="ts">
  /**
   * The served next-step buttons of a completed combine job. They stay
   * disabled, with the target project's served status beside them, until
   * that status is `active` (a step run against a `building` project 409s).
   */
  import { combineLabel } from '$lib/combine/combineText';
  import type { CombineNextStep } from '$lib/types_combine';

  interface Props {
    steps: CombineNextStep[];
    /** The target is served `active`. */
    ready: boolean;
    /** The target's served status; `null` until first read. */
    targetStatus: string | null;
    onpick: (step: CombineNextStep) => void;
  }
  let { steps, ready, targetStatus, onpick }: Props = $props();
</script>

{#each steps as step (step.action)}
  <button
    type="button"
    class="btn"
    title={step.reason}
    disabled={!ready}
    data-testid="combine-next-step-{step.action}"
    onclick={() => onpick(step)}>{combineLabel(step.action)}</button
  >
{/each}
{#if steps.length > 0 && !ready}
  <span class="text-xs text-amber-300" data-testid="combine-next-step-target-status"
    >Target project: {combineLabel(targetStatus)}</span
  >
{/if}
