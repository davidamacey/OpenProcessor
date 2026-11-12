<script lang="ts">
  /**
   * The served loop under any empty Workspace view, so every view (Clusters,
   * Detections, Gallery) shows the same way forward. Renders nothing until
   * the steps are served.
   */
  import NextStepsCard from '$lib/components/NextStepsCard.svelte';
  import { nextStepsStore } from '$stores/nextSteps.svelte';
  import { projectsStore } from '$stores/projects.svelte';
</script>

{#if nextStepsStore.steps.length > 0 && projectsStore.current}
  <div class="mx-auto max-w-2xl p-4" data-testid="empty-next-steps">
    <NextStepsCard
      steps={nextStepsStore.steps}
      next={nextStepsStore.next}
      slug={projectsStore.current.slug}
    />
  </div>
{/if}
