<script lang="ts">
  import Button from '$components/ui/Button.svelte';
  import Icon from '$components/ui/Icon.svelte';
  /**
   * The served loop: every step in the order the backend gives, the first
   * undone one marked as the next. Label, detail, done flag and the page each
   * link opens all come from `GET /next_steps`; nothing is derived here.
   */
  import { resolve } from '$app/paths';
  import { projectHref, type ProjectSubPath } from '$lib/projectPaths';
  import type { WorkflowStep } from '$lib/types_workflow';

  let {
    steps,
    next,
    slug,
    onaction,
  }: {
    steps: readonly WorkflowStep[];
    next: WorkflowStep | null;
    slug: string;
    /** A step's served `action` (a write instead of a page): the page opens its dialog. */
    onaction?: (step: WorkflowStep) => void;
  } = $props();

  /** The button's words: the step's served label (`Create classes from the detector`). */
  const actionLabel = (step: WorkflowStep): string => step.label;

  function path(step: WorkflowStep): ProjectSubPath {
    return `/${step.section}${step.query ? `?${step.query}` : ''}` as ProjectSubPath;
  }
</script>

<section class="surface p-4" data-testid="next-steps" aria-labelledby="next-steps-title">
  <h2 id="next-steps-title" class="mb-3 text-title font-semibold">Next steps</h2>
  <ol class="flex flex-col gap-2">
    {#each steps as step (step.id)}
      <li
        class="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-body"
        data-testid="next-step"
        data-step-id={step.id}
        data-done={step.done}
      >
        <Icon
          name={step.done ? 'circle-check' : 'circle'}
          size={14}
          class={step.done ? 'text-success-text' : 'text-fg-3'}
          label={step.done ? 'Done' : 'To do'}
        />
        <a
          href={resolve(projectHref(path(step), slug))}
          class="font-medium {step.done ? 'text-fg-2' : 'text-fg-1'}"
          aria-current={next?.id === step.id ? 'step' : undefined}
          data-testid={next?.id === step.id ? 'next-step-current' : undefined}
          >{step.label}</a
        >
        <span class="text-fg-2">{step.detail}</span>
        {#if step.action && !step.done && onaction}
          <Button
            size="sm"
            variant={next?.id === step.id ? 'primary' : 'secondary'}
            onclick={() => onaction(step)}
            data-testid="next-step-action">{actionLabel(step)}</Button
          >
        {/if}
      </li>
    {/each}
  </ol>
</section>
