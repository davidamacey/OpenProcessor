<script lang="ts">
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
  }: { steps: readonly WorkflowStep[]; next: WorkflowStep | null; slug: string } =
    $props();

  function path(step: WorkflowStep): ProjectSubPath {
    return `/${step.section}${step.query ? `?${step.query}` : ''}` as ProjectSubPath;
  }
</script>

<section class="surface p-4" data-testid="next-steps" aria-labelledby="next-steps-title">
  <h2 id="next-steps-title" class="mb-3 text-lg font-semibold">Next steps</h2>
  <ol class="flex flex-col gap-2">
    {#each steps as step (step.id)}
      <li
        class="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-sm"
        data-testid="next-step"
        data-step-id={step.id}
        data-done={step.done}
      >
        <span
          class={step.done ? 'text-emerald-400' : 'text-zinc-500'}
          aria-label={step.done ? 'Done' : 'To do'}>{step.done ? '✓' : '○'}</span
        >
        <a
          href={resolve(projectHref(path(step), slug))}
          class="font-medium {step.done ? 'text-zinc-400' : 'text-zinc-100'}"
          aria-current={next?.id === step.id ? 'step' : undefined}
          data-testid={next?.id === step.id ? 'next-step-current' : undefined}
          >{step.label}</a
        >
        <span class="text-zinc-400">{step.detail}</span>
      </li>
    {/each}
  </ol>
</section>
