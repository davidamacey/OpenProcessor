<script lang="ts">
  /**
   * The audit outcomes (the human agreed, the detector was wrong, the VLM was wrong, both were
   * wrong). Each served count opens Detections on those items (`audit_outcome=<value>`).
   */
  import { resolve } from '$app/paths';
  import { humanizeId } from '$lib/humanizeId';
  import { projectHref, type ProjectSubPath } from '$lib/projectPaths';

  let { outcomes }: { outcomes: Record<string, number> } = $props();

  const href = (outcome: string): string =>
    projectHref(
      `/workspace?view=detections&audit_outcome=${encodeURIComponent(outcome)}` as ProjectSubPath,
    );
</script>

<ul class="flex flex-wrap gap-2 text-meta" data-testid="audit-outcomes">
  {#each Object.entries(outcomes) as [outcome, n] (outcome)}
    <li>
      <a
        class="chip hover:border-accent hover:text-accent-text"
        href={resolve(href(outcome) as '/')}
        data-outcome={outcome}
        title="Open these items in Detections"
      >
        {humanizeId(outcome)}: <span class="tabular-nums">{n}</span>
      </a>
    </li>
  {/each}
</ul>
